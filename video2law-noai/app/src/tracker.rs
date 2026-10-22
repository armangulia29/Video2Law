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
