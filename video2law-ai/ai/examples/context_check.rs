//! Context check: replays the request that failed before (a glowing-ball clip and a
//! multi-part teaching request). The AI must see the app state, recognise the ball,
//! call track_object on it, and then write the teaching plan instead of refusing.
//! Run with: cargo run -p ai --example context_check

use std::time::Instant;

use ai::{Action, ActionResult, FromAi, ToAi};

const REQUEST: &str = "Please track the motion of this glowing ball accurately and make a plan to \
impart SHM and Waves and AC all through this to High School Students. Use Law Discovery and Neptune \
Mode to add some flair to it and make it a bit interesting.";

fn main() {
    let started = Instant::now();
    let (to_ai, from_ai) = ai::start();
    for event in from_ai {
        let t = started.elapsed().as_secs_f32();
        match event {
            FromAi::Loading(text) => println!("[{t:5.1}s] {text}"),
            FromAi::Ready => {
                println!("[{t:5.1}s] READY\n[{t:5.1}s] YOU: {REQUEST}");
                let _ = to_ai.send(ToAi::Said(REQUEST.into()));
            }
            FromAi::Do(action, reply) => {
                println!("[{t:5.1}s]   action: {}", short(&action));
                let _ = reply.send(fake_app(action));
            }
            FromAi::Reply(text) => {
                println!("[{t:5.1}s] AI: {text}");
                break;
            }
            FromAi::Error(text) => {
                println!("ERROR: {text}");
                break;
            }
            _ => {}
        }
    }
}

fn short(action: &Action) -> String {
    let s = format!("{action:?}");
    if s.len() > 120 { format!("{}…", &s[..120]) } else { s }
}

/// A stand-in app with the glowing-ball clip open (not yet tracked).
fn fake_app(action: Action) -> ActionResult {
    match action {
        Action::GetResults => ActionResult::Text(
            "Video: Fourier_Epicycles_Animation.mp4 (471 frames, 7.8 s).\nMarker: not set. Scale: NOT set, so g is meaningless.\nTracked frames: 0.\nLaw: not discovered yet\n".into(),
        ),
        Action::GetFrame => ActionResult::Frame(ball_frame(260.0, 120.0)),
        Action::GetKeyframes { count } => {
            ActionResult::Frames((0..count.min(4)).map(|k| (k * 100, ball_frame(200.0 + 30.0 * k as f32, 120.0))).collect())
        }
        Action::UseAiTrack { boxes } => ActionResult::Text(format!("Tracked 471 frames from {} AI sightings.", boxes.len())),
        Action::DiscoverLaw => ActionResult::Text(
            "Law: θ̈ = −24.1·sin θ − 0.05·θ̇ (g ≈ 9.7 m/s², L = 0.40 m)\nConserved quantity: E = ½θ̇² − 24.0 cos θ\nEnergy leak: dE/dt = −0.05·θ̇²".into(),
        ),
        other => ActionResult::Text(format!("Done: {other:?}")),
    }
}

/// A dark 480×270 frame with a glowing yellow ball (radius 18, soft halo) at (x, y).
fn ball_frame(x: f32, y: f32) -> image::RgbImage {
    let mut img = image::RgbImage::from_pixel(480, 270, image::Rgb([12, 10, 8]));
    for py in 0..270 {
        for px in 0..480 {
            let d = (px as f32 - x).hypot(py as f32 - y);
            let glow = if d < 18.0 { 1.0 } else { (1.0 - (d - 18.0) / 60.0).max(0.0) * 0.35 };
            if glow > 0.0 {
                img.put_pixel(px, py, image::Rgb([(250.0 * glow) as u8 + 5, (210.0 * glow) as u8 + 5, (90.0 * glow) as u8]));
            }
        }
    }
    img
}
