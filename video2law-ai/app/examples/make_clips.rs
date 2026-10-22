//! Make realistic demo clips from Rapier scenes (stage backup + test material).
//! Run with: cargo run -p app --release --example make_clips
//!
//! Writes into clips/: synthetic_two_cards.mp4, synthetic_hand.mp4,
//! synthetic_swing_1080p.mp4 and a README.txt with the true values.

use std::io::Write;
use std::process::{Command, Stdio};

const BACKGROUND: u8 = 235;
const GREY: [u8; 3] = [70, 70, 70];
const RED: [u8; 3] = [220, 30, 30];
const BLUE: [u8; 3] = [40, 80, 220];
const SKIN: [u8; 3] = [230, 190, 160];

enum Shape {
    Disc { x: f64, y: f64, r: f64, rgb: [u8; 3] },
    Line { from: (f64, f64), to: (f64, f64), rgb: [u8; 3] },
    Rect { x: f64, y: f64, w: f64, h: f64, rgb: [u8; 3] }, // centred at (x, y)
}

impl Shape {
    /// Pixel box that contains the shape (only these pixels are tested).
    fn bounds(&self) -> (f64, f64, f64, f64) {
        match self {
            Shape::Disc { x, y, r, .. } => (x - r, y - r, x + r, y + r),
            Shape::Line { from, to, .. } => (from.0.min(to.0) - 2.0, from.1.min(to.1) - 2.0, from.0.max(to.0) + 2.0, from.1.max(to.1) + 2.0),
            Shape::Rect { x, y, w, h, .. } => (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0),
        }
    }

    fn covers(&self, px: f64, py: f64) -> Option<[u8; 3]> {
        let inside = match self {
            Shape::Disc { x, y, r, .. } => (px - x).hypot(py - y) < *r,
            Shape::Line { from, to, .. } => {
                let (dx, dy) = (to.0 - from.0, to.1 - from.1);
                let s = (((px - from.0) * dx + (py - from.1) * dy) / (dx * dx + dy * dy).max(1e-9)).clamp(0.0, 1.0);
                (px - from.0 - s * dx).hypot(py - from.1 - s * dy) < 1.5
            }
            Shape::Rect { x, y, w, h, .. } => (px - x).abs() < w / 2.0 && (py - y).abs() < h / 2.0,
        };
        let rgb = match self {
            Shape::Disc { rgb, .. } | Shape::Line { rgb, .. } | Shape::Rect { rgb, .. } => *rgb,
        };
        inside.then_some(rgb)
    }
}

/// Encode `frames` frames of size w×h at 30 fps, drawing `shapes_at(frame)` on a plain wall.
fn film(path: &str, w: usize, h: usize, frames: usize, shapes_at: impl Fn(usize) -> Vec<Shape>) {
    let mut ffmpeg = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", "30", "-i", "-"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path])
        .stdin(Stdio::piped())
        .spawn()
        .expect("ffmpeg on PATH");
    let stdin = ffmpeg.stdin.as_mut().unwrap();
    for f in 0..frames {
        let mut img = vec![BACKGROUND; w * h * 3];
        for shape in shapes_at(f) {
            let (x0, y0, x1, y1) = shape.bounds();
            for py in (y0.max(0.0) as usize)..=(y1.min(h as f64 - 1.0) as usize) {
                for px in (x0.max(0.0) as usize)..=(x1.min(w as f64 - 1.0) as usize) {
                    if let Some(rgb) = shape.covers(px as f64, py as f64) {
                        img[(py * w + px) * 3..][..3].copy_from_slice(&rgb);
                    }
                }
            }
        }
        stdin.write_all(&img).unwrap();
    }
    drop(ffmpeg.stdin.take());
    ffmpeg.wait().unwrap();
    let mb = std::fs::metadata(path).map_or(0.0, |m| m.len() as f64 / 1e6);
    println!("wrote {path} ({mb:.1} MB)");
}

fn main() {
    // Metres (y up) → pixels (y down) around a pixel origin.
    let px = |origin: (f64, f64), scale: f64, x: f64, y: f64| (origin.0 + scale * x, origin.1 - scale * y);

    // 1. Two ID cards on one cable (Neptune Setup A), 600 px per metre.
    let (left, right) = engine::testdata::coupled_pendulums(0.5, 0.3, 3.0, 0.3, 40.0);
    let (lp, rp) = ((450.0, 60.0), (630.0, 60.0));
    film("clips/synthetic_two_cards.mp4", 1280, 720, left.t.len(), |f| {
        let l = px(lp, 600.0, left.x[f], left.y[f]);
        let r = px(lp, 600.0, right.x[f], right.y[f]); // right.x already includes the 0.3 m spacing
        vec![
            Shape::Line { from: (lp.0 - 40.0, 60.0), to: (rp.0 + 40.0, 60.0), rgb: GREY },
            Shape::Line { from: lp, to: l, rgb: GREY },
            Shape::Line { from: rp, to: r, rgb: GREY },
            Shape::Disc { x: l.0, y: l.1, r: 14.0, rgb: RED },
            Shape::Disc { x: r.0, y: r.1, r: 14.0, rgb: BLUE },
        ]
    });

    // 2. Keys on a hidden hand (Neptune Setup B), 600 px per metre.
    let (keys, hand) = engine::testdata::hidden_hand(0.5, 3.0, 20.0);
    let origin = (640.0, 60.0);
    film("clips/synthetic_hand.mp4", 1280, 720, keys.t.len(), |f| {
        let hnd = px(origin, 600.0, hand.x[f], hand.y[f]);
        let k = px(origin, 600.0, keys.x[f], keys.y[f]);
        vec![
            Shape::Line { from: hnd, to: k, rgb: GREY },
            Shape::Rect { x: hnd.0, y: hnd.1, w: 60.0, h: 40.0, rgb: SKIN },
            Shape::Disc { x: k.0, y: k.1, r: 14.0, rgb: RED },
        ]
    });

    // 3. A long damped swing in full HD (tests memory use and shrinking), 900 px per metre.
    let swing = engine::testdata::pendulum(0.5, 0.8, 0.3, 30.0);
    let pivot = (960.0, 80.0);
    film("clips/synthetic_swing_1080p.mp4", 1920, 1080, swing.t.len(), |f| {
        let b = px(pivot, 900.0, swing.x[f], swing.y[f]);
        vec![Shape::Line { from: pivot, to: b, rgb: GREY }, Shape::Disc { x: b.0, y: b.1, r: 14.0, rgb: RED }]
    });

    std::fs::write(
        "clips/README.txt",
        "Synthetic clips made from Rapier physics (cargo run -p app --release --example make_clips).\n\
         True values: lanyard L = 0.5 m, g = 9.81 m/s^2.\n\n\
         synthetic_two_cards.mp4  1280x720, 40 s. Neptune Setup A: red (left) card released, blue (right) card starts still,\n\
           joined by a soft cable (coupling 3.0). 600 px per metre in the original video; pivots 0.3 m (180 px) apart.\n\
           Track the red sticker, crop to the left half, set the hidden pivot at the top of the blue lanyard.\n\
         synthetic_hand.mp4       1280x720, 20 s. Neptune Setup B: keys (red) hang from a hand (skin box) that is still\n\
           for the first 3 s, then sways. 600 px per metre.\n\
         synthetic_swing_1080p.mp4 1920x1080, 30 s. Damped swing (friction 0.3, released at 0.8 rad). 900 px per metre.\n\
         synthetic_swing.mp4 / synthetic_two_cards_small.mp4: 640x480 versions from the e2e_video check, 400 px per metre.\n",
    )
    .unwrap();
    println!("wrote clips/README.txt");
}
