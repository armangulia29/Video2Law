use eframe::egui::{Color32, Painter, Pos2, Rect, Stroke};
use engine::Trajectory;

/// Draws tracking overlays directly over the displayed video frame.
pub struct FrameOverlay<'a> {
    pub painter: &'a Painter,
    pub image_rect: Rect,
    pub frame_width: f32,
    pub frame_height: f32,
}

impl<'a> FrameOverlay<'a> {
    pub fn new(
        painter: &'a Painter,
        image_rect: Rect,
        frame_width: usize,
        frame_height: usize,
    ) -> Self {
        Self {
            painter,
            image_rect,
            frame_width: frame_width as f32,
            frame_height: frame_height as f32,
        }
    }

    /// Converts pixel coordinate (x, y) in the source video into screen Pos2.
    pub fn video_to_screen(&self, x: f32, y: f32) -> Pos2 {
        let scale_x = self.image_rect.width() / self.frame_width.max(1.0);
        let scale_y = self.image_rect.height() / self.frame_height.max(1.0);
        Pos2::new(
            self.image_rect.min.x + x * scale_x,
            self.image_rect.min.y + y * scale_y,
        )
    }

    /// Draws the currently tracked marker dot with a target ring.
    pub fn draw_current_marker(&self, px: f64, py: f64) {
        let center = self.video_to_screen(px as f32, py as f32);

        // Outer glowing ring
        self.painter.circle_stroke(
            center,
            10.0,
            Stroke::new(2.5, Color32::from_rgb(0, 255, 200)),
        );
        // Center dot
        self.painter.circle_filled(
            center,
            4.0,
            Color32::from_rgb(255, 255, 0),
        );
    }

    /// Draws the historical trajectory path up to the current frame.
    pub fn draw_trajectory_trail(&self, points: &[Option<[f64; 2]>], current_frame: usize) {
        if points.is_empty() {
            return;
        }

        let end = current_frame.min(points.len().saturating_sub(1));
        let mut prev_pos: Option<Pos2> = None;

        for (i, opt_pt) in points[..=end].iter().enumerate() {
            if let Some([px, py]) = *opt_pt {
                let screen_pos = self.video_to_screen(px as f32, py as f32);
                if let Some(prev) = prev_pos {
                    // Fade trail older than 60 frames
                    let age = end - i;
                    let alpha = (255.0 * (1.0 - (age as f32 / 60.0).clamp(0.0, 0.85))) as u8;
                    self.painter.line_segment(
                        [prev, screen_pos],
                        Stroke::new(1.8, Color32::from_rgba_unmultiplied(0, 220, 255, alpha)),
                    );
                }
                prev_pos = Some(screen_pos);
            } else {
                prev_pos = None;
            }
        }
    }

    /// Draws scale calibration points and the connecting reference line.
    pub fn draw_scale_calibration(
        &self,
        p1: Option<[f64; 2]>,
        p2: Option<[f64; 2]>,
        dist_mm: f64,
    ) {
        let stroke = Stroke::new(2.0, Color32::from_rgb(255, 160, 40));

        let sp1 = p1.map(|[x, y]| self.video_to_screen(x as f32, y as f32));
        let sp2 = p2.map(|[x, y]| self.video_to_screen(x as f32, y as f32));

        if let (Some(pt1), Some(pt2)) = (sp1, sp2) {
            self.painter.line_segment([pt1, pt2], stroke);
            let mid = Pos2::new((pt1.x + pt2.x) * 0.5, (pt1.y + pt2.y) * 0.5 - 12.0);
            self.painter.text(
                mid,
                eframe::egui::Align2::CENTER_CENTER,
                format!("{:.1} mm", dist_mm * 1000.0),
                eframe::egui::FontId::proportional(13.0),
                Color32::from_rgb(255, 180, 50),
            );
        }

        if let Some(pt1) = sp1 {
            self.painter.circle_filled(pt1, 5.0, Color32::from_rgb(255, 80, 0));
        }
        if let Some(pt2) = sp2 {
            self.painter.circle_filled(pt2, 5.0, Color32::from_rgb(255, 80, 0));
        }
    }

    /// Where `traj` is at `time`, converted from metres (y up) back to video pixels
    /// around `anchor`. None if the trajectory doesn't cover that moment.
    fn pixel_at(traj: &Trajectory, time: f64, anchor: [f64; 2], m_per_px: f64) -> Option<(f32, f32)> {
        if m_per_px <= 0.0 {
            return None;
        }
        let i = traj.t.iter().enumerate().min_by(|a, b| (a.1 - time).abs().total_cmp(&(b.1 - time).abs()))?.0;
        if (traj.t[i] - time).abs() > 0.2 {
            return None;
        }
        Some(((anchor[0] + traj.x[i] / m_per_px) as f32, (anchor[1] - traj.y[i] / m_per_px) as f32))
    }

    /// Draws the predicted "ghost" object on screen.
    pub fn draw_ghost(&self, ghost: &Trajectory, current_time: f64, origin: [f64; 2], m_per_px: f64) {
        let Some((gx, gy)) = Self::pixel_at(ghost, current_time, origin, m_per_px) else { return };
        let screen_pos = self.video_to_screen(gx, gy);

        // Draw translucent ghost card
        self.painter.circle_stroke(
            screen_pos,
            14.0,
            Stroke::new(2.0, Color32::from_rgba_unmultiplied(255, 80, 150, 200)),
        );
        self.painter.circle_filled(
            screen_pos,
            6.0,
            Color32::from_rgba_unmultiplied(255, 120, 180, 150),
        );
    }

    /// Neptune Mode: darken the part of the frame the camera "can't see"
    /// (everything right of `keep_left` × width).
    pub fn draw_crop(&self, keep_left: f64) {
        if keep_left >= 1.0 {
            return;
        }
        let x = self.video_to_screen((keep_left as f32) * self.frame_width, 0.0).x;
        let hidden = Rect::from_min_max(Pos2::new(x, self.image_rect.min.y), self.image_rect.max);
        self.painter.rect_filled(hidden, 0.0, Color32::from_rgba_unmultiplied(10, 10, 20, 235));
        self.painter.text(
            hidden.center(),
            eframe::egui::Align2::CENTER_CENTER,
            "hidden from the camera",
            eframe::egui::FontId::proportional(16.0),
            Color32::from_gray(140),
        );
    }

    /// Neptune Mode: the rebuilt hidden object (card or hand), drawn around `anchor`
    /// (pixels), with its string/lanyard from the anchor for a card.
    pub fn draw_hidden(&self, hidden: &Trajectory, current_time: f64, anchor: [f64; 2], m_per_px: f64, label: &str) {
        let Some((hx, hy)) = Self::pixel_at(hidden, current_time, anchor, m_per_px) else { return };
        let pos = self.video_to_screen(hx, hy);
        let colour = Color32::from_rgb(255, 120, 220);
        if label == "hidden card" {
            let pivot = self.video_to_screen(anchor[0] as f32, anchor[1] as f32);
            self.painter.line_segment([pivot, pos], Stroke::new(1.5, colour.gamma_multiply(0.6)));
        }
        self.painter.circle_stroke(pos, 16.0, Stroke::new(3.0, colour));
        self.painter.circle_filled(pos, 6.0, colour.gamma_multiply(0.7));
        self.painter.text(
            pos + eframe::egui::vec2(0.0, -26.0),
            eframe::egui::Align2::CENTER_CENTER,
            label,
            eframe::egui::FontId::proportional(14.0),
            colour,
        );
    }
}
