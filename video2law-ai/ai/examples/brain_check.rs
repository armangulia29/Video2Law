//! Brain check without the real app: a stand-in "app" answers the AI's actions, and a
//! synthetic frame (red dot on a grey wall) lets us check that the AI can find things.
//! Run with: cargo run -p ai --example brain_check

use std::time::Instant;

use ai::{Action, ActionResult, FromAi, ToAi};

const QUESTIONS: [&str; 3] = [
    "Track the red dot in the video.",
    "What did you find for the law and for g?",
    "Why would the pendulum swing slower on the Moon?",
];

fn main() {
    let started = Instant::now();
    let (to_ai, from_ai) = ai::start();
    let mut asked = 0;

    for event in from_ai {
        match event {
            FromAi::Loading(text) => println!("[{:5.1}s] loading: {text}", started.elapsed().as_secs_f32()),
            FromAi::Ready => {
                println!("[{:5.1}s] READY", started.elapsed().as_secs_f32());
                ask(&to_ai, &mut asked, &started);
            }
            FromAi::Heard(text) => println!("heard: {text}"),
            FromAi::Thinking => {}
            FromAi::Do(action, reply) => {
                println!("[{:5.1}s]   action: {action:?}", started.elapsed().as_secs_f32());
                let _ = reply.send(fake_app(action));
            }
            FromAi::Reply(text) => {
                println!("[{:5.1}s] AI: {text}\n", started.elapsed().as_secs_f32());
                if asked == QUESTIONS.len() {
                    break;
                }
                ask(&to_ai, &mut asked, &started);
            }
            FromAi::Error(text) => {
                println!("ERROR: {text}");
                break;
            }
        }
    }
}

fn ask(to_ai: &std::sync::mpsc::Sender<ToAi>, asked: &mut usize, started: &Instant) {
    println!("[{:5.1}s] YOU: {}", started.elapsed().as_secs_f32(), QUESTIONS[*asked]);
    let _ = to_ai.send(ToAi::Said(QUESTIONS[*asked].to_string()));
    *asked += 1;
}

/// What the real app would answer. The red dot sits at x 300–340, y 200–240.
fn fake_app(action: Action) -> ActionResult {
    match action {
        Action::GetFrame => {
            let mut img = image::RgbImage::from_pixel(480, 270, image::Rgb([200, 200, 200]));
            for y in 200..240 {
                for x in 300..340 {
                    if (x as f32 - 320.0).hypot(y as f32 - 220.0) < 20.0 {
                        img.put_pixel(x, y, image::Rgb([220, 30, 30]));
                    }
                }
            }
            ActionResult::Frame(img)
        }
        Action::GetKeyframes { count } => ActionResult::Frames((0..count.min(6)).map(|k| (k * 10, dot_frame(100.0 + 50.0 * k as f32, 120.0))).collect()),
        Action::UseAiTrack { boxes } => {
            for (i, b) in &boxes {
                let truth = 100.0 + 5.0 * *i as f64;
                let cx = (b[0] + b[2]) / 2.0;
                println!("          keyframe {i}: AI box centre x {cx:.0} (dot at {truth:.0}) {}", if (cx - truth).abs() < 20.0 { "✓" } else { "✗" });
            }
            ActionResult::Text(format!("Tracked 360 frames from {} AI sightings.", boxes.len()))
        }
        Action::UseObject { bbox, use_as, .. } => {
            let (cx, cy) = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0);
            let hit = (cx - 320.0).abs() < 25.0 && (cy - 220.0).abs() < 25.0;
            println!("          box {bbox:.0?} → centre ({cx:.0}, {cy:.0}) {}", if hit { "ON the dot ✓" } else { "MISSED ✗" });
            ActionResult::Text(format!("Marker set ({use_as}) at ({cx:.0}, {cy:.0})."))
        }
        Action::GetResults => ActionResult::Text(
            "Video: synthetic_swing.mp4 (360 frames, 12.0 s). Marker: set. Scale: calibrated.\n\
             Law: θ̈ = −19.59·sin θ − 0.30·θ̇    (g ≈ 9.79 m/s², L = 0.50 m)\n\
             Conserved quantity: E = ½θ̇² − 19.50 cos θ\nEnergy leak: dE/dt = −0.30·θ̇²"
                .into(),
        ),
        other => ActionResult::Text(format!("Done: {other:?}")),
    }
}

/// A grey 480×270 frame with a red dot (radius 15) centred at (x, y).
fn dot_frame(x: f32, y: f32) -> image::RgbImage {
    let mut img = image::RgbImage::from_pixel(480, 270, image::Rgb([200, 200, 200]));
    for py in 0..270 {
        for px in 0..480 {
            if (px as f32 - x).hypot(py as f32 - y) < 15.0 {
                img.put_pixel(px, py, image::Rgb([220, 30, 30]));
            }
        }
    }
    img
}
