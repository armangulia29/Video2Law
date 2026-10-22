use std::sync::Arc;

use crate::source::VideoFrame;
use engine::Trajectory;

/// RGB color picked from a marker pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorRgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl ColorRgb {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Euclidean distance between two colors in RGB space.
    pub fn distance(&self, other: &ColorRgb) -> f64 {
        let dr = self.r as f64 - other.r as f64;
        let dg = self.g as f64 - other.g as f64;
        let db = self.b as f64 - other.b as f64;
        (dr * dr + dg * dg + db * db).sqrt()
    }
}

/// Scale calibration tool: converts screen pixels to real-world meters.
///
/// Default real-world distance is 0.0856 m (85.6 mm), the standard width of an ID card / credit card.
#[derive(Debug, Clone)]
pub struct ScaleTool {
    pub p1: Option<[f64; 2]>,
    pub p2: Option<[f64; 2]>,
    pub real_distance_m: f64,
}

impl Default for ScaleTool {
    fn default() -> Self {
        Self {
            p1: None,
            p2: None,
            real_distance_m: 0.0856, // 85.6 mm (ID card standard)
        }
    }
}

impl ScaleTool {
    /// Distance in pixels between the two calibration points.
    pub fn pixel_distance(&self) -> Option<f64> {
        match (self.p1, self.p2) {
            (Some([x1, y1]), Some([x2, y2])) => {
                let dist = (x2 - x1).hypot(y2 - y1);
                if dist > 1.0 { Some(dist) } else { None }
            }
            _ => None,
        }
    }

    /// True once both scale points are clicked. Without it, g comes out wrong.
    pub fn is_calibrated(&self) -> bool {
        self.pixel_distance().is_some() && self.real_distance_m > 0.0
    }

    /// Scale factor in meters per pixel.
    pub fn meters_per_pixel(&self) -> f64 {
        match self.pixel_distance() {
            Some(px_dist) if self.real_distance_m > 0.0 => self.real_distance_m / px_dist,
            // Uncalibrated fallback (1 px = 1 mm). The UI warns that g is meaningless.
            _ => 0.001,
        }
    }
}

/// Tracks a colored blob marker across video frames to produce an engine::Trajectory.
#[derive(Debug, Clone)]
pub struct Tracker {
    pub target_color: Option<ColorRgb>,
    pub tolerance: f64, // max color distance (default ~35.0)
    pub scale: ScaleTool,
    /// Share of the frame width, from the left, that the camera "sees" (1.0 = all).
    /// Neptune Mode crops the right part away, so a second card there is never tracked.
    pub keep_left: f64,
    pub tracked_points: Vec<Option<[f64; 2]>>, // [pixel_x, pixel_y] per frame
}

impl Default for Tracker {
    fn default() -> Self {
        Self {
            target_color: None,
            tolerance: 35.0,
            scale: ScaleTool::default(),
            keep_left: 1.0,
            tracked_points: Vec::new(),
        }
    }
}

impl Tracker {
    /// Detects the marker centroid in a single frame, looking only at columns `x < x_limit`.
    ///
    /// If `prev_point` is provided, search near it first and fall back to the whole frame.
    pub fn detect_blob(
        frame: &VideoFrame,
        target: ColorRgb,
        tolerance: f64,
        prev_point: Option<[f64; 2]>,
        x_limit: usize,
    ) -> Option<[f64; 2]> {
        let (width, height) = (frame.width, frame.height);
        let max_x = x_limit.min(width).max(1) - 1;

        // Average position of all pixels close to the target colour inside a box.
        let centroid = |min_x: usize, max_x: usize, min_y: usize, max_y: usize| {
            let (mut sum_x, mut sum_y, mut count) = (0.0, 0.0, 0.0);
            for y in min_y..=max_y {
                let row_offset = y * width * 3;
                for x in min_x..=max_x {
                    let idx = row_offset + x * 3;
                    let c = ColorRgb::new(frame.rgb[idx], frame.rgb[idx + 1], frame.rgb[idx + 2]);
                    if target.distance(&c) <= tolerance {
                        sum_x += x as f64;
                        sum_y += y as f64;
                        count += 1.0;
                    }
                }
            }
            (sum_x, sum_y, count)
        };

        // First look in a neighbourhood of the previous point (fast, and ignores look-alikes far away).
        let mut found = (0.0, 0.0, 0.0);
        if let Some([px, py]) = prev_point {
            let radius = 0.2 * width.max(height) as f64;
            found = centroid(
                (px - radius).max(0.0) as usize,
                ((px + radius) as usize).min(max_x),
                (py - radius).max(0.0) as usize,
                ((py + radius) as usize).min(height - 1),
            );
        }
        if found.2 < 4.0 {
            found = centroid(0, max_x, 0, height - 1);
        }
        let (sum_x, sum_y, count) = found;
        if count >= 3.0 { Some([sum_x / count, sum_y / count]) } else { None }
    }

    /// Tracks all frames in sequence and builds a physics Trajectory in meters.
    ///
    /// Positions are relative to the first tracked point, with y pointing up.
    pub fn track_frames(&mut self, frames: &[Arc<VideoFrame>]) -> Trajectory {
        self.tracked_points.clear();
        let Some(target) = self.target_color else { return Trajectory::default() };

        let mut last_detected: Option<[f64; 2]> = None;
        for frame in frames {
            let x_limit = (frame.width as f64 * self.keep_left).round() as usize;
            let pt = Self::detect_blob(frame, target, self.tolerance, last_detected, x_limit);
            if pt.is_some() {
                last_detected = pt;
            }
            self.tracked_points.push(pt);
        }

        self.to_trajectory(frames)
    }

    /// AI tracking. The vision model found the object in some keyframes
    /// (`anchors`: frame index → box). Between them we follow the object by its
    /// *appearance*: each frame, slide the AI-found patch around the expected spot and
    /// keep the best match (normalised cross-correlation, i.e. "which spot looks most
    /// like the object"). At every AI keyframe the patch is refreshed, and if our
    /// track has drifted away from where the AI saw the object, we jump back to it.
    pub fn track_with_ai(&mut self, frames: &[Arc<VideoFrame>], anchors: &[(usize, [f64; 4])]) -> Trajectory {
        self.tracked_points = vec![None; frames.len()];
        let mut anchors = anchors.to_vec();
        anchors.sort_by_key(|a| a.0);
        let Some(&(first, first_box)) = anchors.first() else { return Trajectory::default() };

        let centre = |b: [f64; 4]| [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0];
        let mut template = Patch::cut(&frames[first], first_box);
        let mut pos = centre(first_box);
        let mut velocity = [0.0, 0.0];
        let mut next_anchor = 0;

        // Track forwards from the first AI sighting, then backwards to the start.
        for i in first..frames.len() {
            if next_anchor < anchors.len() && anchors[next_anchor].0 == i {
                let (_, b) = anchors[next_anchor];
                let ai = centre(b);
                let size = (b[2] - b[0]).max(b[3] - b[1]);
                // Trust our smooth track unless it has wandered off the AI's sighting.
                if i == first || (pos[0] - ai[0]).hypot(pos[1] - ai[1]) > 0.5 * size {
                    pos = ai;
                }
                template = Patch::cut(&frames[i], b);
                next_anchor += 1;
            } else {
                let guess = [pos[0] + velocity[0], pos[1] + velocity[1]];
                let found = template.find(&frames[i], guess);
                velocity = [found[0] - pos[0], found[1] - pos[1]];
                pos = found;
            }
            self.tracked_points[i] = Some(pos);
        }
        let mut pos = centre(first_box);
        let template = Patch::cut(&frames[first], first_box);
        for i in (0..first).rev() {
            pos = template.find(&frames[i], pos);
            self.tracked_points[i] = Some(pos);
        }
        self.to_trajectory(frames)
    }

    /// Pixel positions → metres relative to the first tracked point, y pointing up.
    fn to_trajectory(&self, frames: &[Arc<VideoFrame>]) -> Trajectory {
        let Some(origin) = self.tracked_points.iter().find_map(|&p| p) else {
            return Trajectory::default();
        };
        let m_per_px = self.scale.meters_per_pixel();
        let filled = fill_gaps(&self.tracked_points);

        Trajectory {
            t: frames.iter().map(|f| f.time).collect(),
            x: filled.iter().map(|p| (p[0] - origin[0]) * m_per_px).collect(),
            // Invert y so positive y points up in physics space
            y: filled.iter().map(|p| (origin[1] - p[1]) * m_per_px).collect(),
        }
    }
}

/// A greyscale picture of the object, cut from a frame at the AI's box.
struct Patch {
    w: usize,
    h: usize,
    pixels: Vec<f32>, // brightness minus its average, so lighting changes don't matter
    norm: f32,
}

impl Patch {
    fn cut(frame: &VideoFrame, b: [f64; 4]) -> Patch {
        let w = ((b[2] - b[0]) as usize).clamp(6, 90);
        let h = ((b[3] - b[1]) as usize).clamp(6, 90);
        let (x0, y0) = ((b[0] + b[2]) / 2.0 - w as f64 / 2.0, (b[1] + b[3]) / 2.0 - h as f64 / 2.0);
        let mut pixels: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| grey(frame, x0 as isize + x as isize, y0 as isize + y as isize))
            .collect();
        let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
        pixels.iter_mut().for_each(|p| *p -= mean);
        let norm = pixels.iter().map(|p| p * p).sum::<f32>().sqrt().max(1e-6);
        Patch { w, h, pixels, norm }
    }

    /// The centre of the spot near `guess` that looks most like this patch.
    fn find(&self, frame: &VideoFrame, guess: [f64; 2]) -> [f64; 2] {
        let radius = (self.w.max(self.h) / 2 + 6) as isize;
        let step = if self.w * self.h > 900 { 2 } else { 1 }; // skip pixels on big patches (speed)
        let (mut best, mut best_score) = (guess, f32::MIN);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let cx = guess[0] + dx as f64;
                let cy = guess[1] + dy as f64;
                let score = self.score(frame, cx, cy, step);
                if score > best_score {
                    best_score = score;
                    best = [cx, cy];
                }
            }
        }
        best
    }

    /// Normalised cross-correlation of the patch centred at (cx, cy): 1 = identical look.
    fn score(&self, frame: &VideoFrame, cx: f64, cy: f64, step: usize) -> f32 {
        let (x0, y0) = ((cx - self.w as f64 / 2.0) as isize, (cy - self.h as f64 / 2.0) as isize);
        let mut window = Vec::with_capacity(self.pixels.len() / (step * step) + 1);
        let mut weights = Vec::with_capacity(window.capacity());
        for y in (0..self.h).step_by(step) {
            for x in (0..self.w).step_by(step) {
                window.push(grey(frame, x0 + x as isize, y0 + y as isize));
                weights.push(self.pixels[y * self.w + x]);
            }
        }
        let mean = window.iter().sum::<f32>() / window.len() as f32;
        let (mut dot, mut norm) = (0.0, 0.0);
        for (v, t) in window.iter().zip(&weights) {
            dot += (v - mean) * t;
            norm += (v - mean) * (v - mean);
        }
        let template_norm = weights.iter().map(|t| t * t).sum::<f32>().sqrt().max(1e-6);
        let _ = self.norm;
        dot / (norm.sqrt().max(1e-6) * template_norm)
    }
}

/// Brightness of a pixel (0–255); pixels outside the frame count as black.
fn grey(frame: &VideoFrame, x: isize, y: isize) -> f32 {
    if x < 0 || y < 0 || x as usize >= frame.width || y as usize >= frame.height {
        return 0.0;
    }
    let i = (y as usize * frame.width + x as usize) * 3;
    0.299 * frame.rgb[i] as f32 + 0.587 * frame.rgb[i + 1] as f32 + 0.114 * frame.rgb[i + 2] as f32
}

/// Fill frames where the sticker wasn't found by drawing a straight line between the
/// nearest found frames before and after. (Repeating the last position instead would
/// make the object look like it stopped dead, which ruins the acceleration.)
/// Missing frames at the very start or end copy the nearest found frame.
fn fill_gaps(points: &[Option<[f64; 2]>]) -> Vec<[f64; 2]> {
    let found: Vec<(usize, [f64; 2])> = points.iter().enumerate().filter_map(|(i, p)| p.map(|p| (i, p))).collect();
    (0..points.len())
        .map(|i| {
            let after = found.partition_point(|&(j, _)| j < i);
            match (after.checked_sub(1).map(|k| found[k]), found.get(after)) {
                (_, Some(&(j, p))) if j == i => p,
                (Some((j0, p0)), Some(&(j1, p1))) => {
                    let s = (i - j0) as f64 / (j1 - j0) as f64;
                    [p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1])]
                }
                (Some((_, p)), None) => p,
                (None, Some(&(_, p))) => p,
                (None, None) => [0.0, 0.0],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_are_bridged_with_straight_lines() {
        let pts = [None, Some([0.0, 0.0]), None, None, Some([3.0, 6.0]), None];
        let filled = fill_gaps(&pts);
        assert_eq!(filled, vec![[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [3.0, 6.0]]);
    }
}
