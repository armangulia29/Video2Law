//! End-to-end check without a camera: film Rapier scenes into real .mp4 files, then run
//! the app's own video reader + tracker + the engine on them, and compare with the truth.
//! Run with: cargo run -p app --release --example e2e_video
//!
//! 1. A damped swing (law, g, ghost).
//! 2. Neptune Setup A: two cards, the right one cropped away, rebuilt from the left one.

#[path = "../src/source.rs"]
mod source;
#[path = "../src/tracker.rs"]
mod tracker;

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use engine::{NeptuneMode, Options, Trajectory};

const W: usize = 640;
const H: usize = 480;
const PX_PER_M: f64 = 400.0; // in the filmed video

/// A filled disc or a line segment, in video pixels.
enum Shape {
    Disc { x: f64, y: f64, r: f64, rgb: [u8; 3] },
    Line { from: (f64, f64), to: (f64, f64), rgb: [u8; 3] },
}

/// Film a scene: `shapes_at(frame)` lists what to draw in that frame. Encoded as H.264.
fn film(path: &str, frames: usize, shapes_at: impl Fn(usize) -> Vec<Shape>) {
    let mut ffmpeg = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{W}x{H}"), "-r", "30", "-i", "-"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", path])
        .stdin(Stdio::piped())
        .spawn()
        .expect("ffmpeg on PATH");
    let stdin = ffmpeg.stdin.as_mut().unwrap();
    for f in 0..frames {
        let mut img = vec![235u8; W * H * 3];
        for shape in shapes_at(f) {
            for py in 0..H {
                for px in 0..W {
                    let (x, y) = (px as f64, py as f64);
                    let (inside, rgb) = match &shape {
                        Shape::Disc { x: cx, y: cy, r, rgb } => ((x - cx).hypot(y - cy) < *r, rgb),
                        Shape::Line { from, to, rgb } => {
                            let (dx, dy) = (to.0 - from.0, to.1 - from.1);
                            let s = (((x - from.0) * dx + (y - from.1) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
                            ((x - from.0 - s * dx).hypot(y - from.1 - s * dy) < 1.5, rgb)
                        }
                    };
                    if inside {
                        img[(py * W + px) * 3..][..3].copy_from_slice(rgb);
                    }
                }
            }
        }
        stdin.write_all(&img).unwrap();
    }
    drop(ffmpeg.stdin.take());
    ffmpeg.wait().unwrap();
}

/// Read a clip back with the app's own reader, exactly like the app does.
fn decode(path: &str) -> (source::VideoMeta, Vec<Arc<source::VideoFrame>>) {
    let meta = source::probe_video(std::path::Path::new(path)).unwrap();
    let (rx, _cancel) = source::start_decoding(path.into(), meta);
    (meta, rx.iter().map(Arc::new).collect())
}

/// A tracker set up like a user would: sticker colour clicked, scale from two points.
fn tracker_for(meta: &source::VideoMeta, rgb: [u8; 3]) -> tracker::Tracker {
    let mut t = tracker::Tracker::default();
    t.target_color = Some(tracker::ColorRgb::new(rgb[0], rgb[1], rgb[2]));
    let px_per_m = PX_PER_M * meta.width as f64 / W as f64; // frames are shrunk on decode
    t.scale.p1 = Some([0.0, 0.0]);
    t.scale.p2 = Some([px_per_m, 0.0]);
    t.scale.real_distance_m = 1.0;
    t
}

const RED: [u8; 3] = [220, 30, 30];
const BLUE: [u8; 3] = [40, 80, 220];
const GREY: [u8; 3] = [90, 90, 90];

fn main() {
    // ── 1. Damped swing ──────────────────────────────────────────────────────────
    let pivot = (320.0, 40.0);
    let to_px = |p: (f64, f64), x: f64, y: f64| (p.0 + PX_PER_M * x, p.1 - PX_PER_M * y);
    let truth = engine::testdata::pendulum(0.5, 0.8, 0.3, 12.0);
    film("clips/synthetic_swing.mp4", truth.t.len(), |f| {
        let (bx, by) = to_px(pivot, truth.x[f], truth.y[f]);
        vec![Shape::Line { from: pivot, to: (bx, by), rgb: GREY }, Shape::Disc { x: bx, y: by, r: 10.0, rgb: RED }]
    });
    let (meta, frames) = decode("clips/synthetic_swing.mp4");
    println!("swing: decoded {} frames, shrunk to {}x{}", frames.len(), meta.width, meta.height);
    let mut tr = tracker_for(&meta, RED);
    let traj = tr.track_frames(&frames);
    let report = engine::analyse(&traj, &Options::default());
    println!("  law:  {}", report.law_plain);
    println!("  E:    {:?}   leak: {:?}", report.conserved_typst, report.leak_typst);
    let shrink = meta.width as f64 / W as f64;
    let origin = tr.tracked_points.iter().find_map(|p| *p).unwrap();
    let worst = worst_miss(&report.ghost, |t| {
        let j = (t * 30.0).round() as usize;
        (j < truth.t.len()).then(|| {
            let (x, y) = to_px(pivot, truth.x[j], truth.y[j]);
            (x * shrink, y * shrink)
        })
    }, origin, tr.scale.meters_per_pixel());
    println!("  ghost vs real card: worst {:.1} cm", worst * 100.0 / (PX_PER_M * shrink));

    // ── 2. Neptune Setup A: two cards, right half cropped ────────────────────────
    let (lp, rp) = ((220.0, 40.0), (220.0 + 0.3 * PX_PER_M, 40.0)); // pivots 30 cm apart
    let (left, right) = engine::testdata::coupled_pendulums(0.5, 0.3, 3.0, 0.3, 40.0);
    film("clips/synthetic_two_cards_small.mp4", left.t.len(), |f| {
        let (lx, ly) = to_px(lp, left.x[f], left.y[f]);
        let (rx, ry) = to_px(lp, right.x[f], right.y[f]); // right.x already includes the 0.3 m offset
        vec![
            Shape::Line { from: lp, to: rp, rgb: GREY },
            Shape::Line { from: lp, to: (lx, ly), rgb: GREY },
            Shape::Line { from: rp, to: (rx, ry), rgb: GREY },
            Shape::Disc { x: lx, y: ly, r: 10.0, rgb: RED },
            Shape::Disc { x: rx, y: ry, r: 10.0, rgb: BLUE },
        ]
    });
    let (meta, frames) = decode("clips/synthetic_two_cards_small.mp4");
    let mut tr = tracker_for(&meta, RED);
    tr.keep_left = 0.5; // the camera only sees the left half
    let traj = tr.track_frames(&frames);
    let opts = Options { neptune: NeptuneMode::CoupledPair, ..Default::default() };
    let report = engine::analyse(&traj, &opts);
    println!("two cards: hidden objects detected = {}, top singular values {:.3?}", report.hidden_count, &report.singular_values[..5]);
    let hidden_pivot = (rp.0 * meta.width as f64 / W as f64, rp.1 * meta.width as f64 / W as f64); // user clicks it
    let shrink = meta.width as f64 / W as f64;
    let worst = worst_miss(report.hidden.as_ref().unwrap(), |t| {
        let j = (t * 30.0).round() as usize;
        (j < right.t.len()).then(|| {
            let (x, y) = to_px(lp, right.x[j], right.y[j]);
            (x * shrink, y * shrink)
        })
    }, [hidden_pivot.0, hidden_pivot.1], tr.scale.meters_per_pixel());
    println!("  hidden card vs real (never seen) card: worst {:.1} cm", worst * 100.0 / (PX_PER_M * shrink));
}

/// Largest distance (in shrunk pixels) between a trajectory drawn around `anchor` and
/// where the real object was, the way the app's overlay draws it.
fn worst_miss(traj: &Trajectory, real_px: impl Fn(f64) -> Option<(f64, f64)>, anchor: [f64; 2], m_per_px: f64) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..traj.t.len() {
        let Some((rx, ry)) = real_px(traj.t[i]) else { break };
        let (gx, gy) = (anchor[0] + traj.x[i] / m_per_px, anchor[1] - traj.y[i] / m_per_px);
        worst = worst.max((gx - rx).hypot(gy - ry));
    }
    worst
}
