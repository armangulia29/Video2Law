//! Full-control check: everyday commands, one at a time. For each, the AI must press
//! the right app button (action). A stand-in app records what was pressed.
//! Run with: cargo run -p ai --example control_check

use std::time::Instant;

use ai::{Action, ActionResult, FromAi, ToAi};

/// (command, words that must appear in one of the actions it triggers)
const COMMANDS: [(&str, &str); 8] = [
    ("Play the video.", "Play { playing: true"),
    ("Pause it and go to 3 seconds.", "Seek"),
    ("What if this happened on the Moon?", "gravity_scale: Some(0.16"),
    ("Remove all friction.", "friction_scale: Some(0.0"),
    ("Make the law simpler: set the sparsity to 0.5.", "SetSparsity { lambda: 0.5"),
    ("Turn on Neptune Mode for two coupled cards and hide the right half of the frame.", "SetNeptune { mode: \"coupled_pair\""),
    ("Now reveal the hidden part.", "Reveal { show: true"),
    ("Stop reading your replies out loud.", "SetSpeakReplies { on: false"),
];

fn main() {
    let started = Instant::now();
    let (to_ai, from_ai) = ai::start();
    let mut asked = 0;
    let mut pressed: Vec<String> = Vec::new();
    let mut passed = 0;

    for event in from_ai {
        let t = started.elapsed().as_secs_f32();
        match event {
            FromAi::Ready => {
                println!("[{t:5.1}s] READY");
                ask(&to_ai, asked);
            }
            FromAi::Do(action, reply) => {
                let s = format!("{action:?}");
                if !matches!(action, Action::GetResults | Action::GetFrame) {
                    println!("[{t:5.1}s]   pressed: {s}");
                }
                pressed.push(s);
                let _ = reply.send(fake_app(action));
            }
            FromAi::Reply(text) => {
                let (_, must) = COMMANDS[asked];
                let ok = pressed.iter().any(|p| p.contains(must));
                passed += ok as usize;
                println!("[{t:5.1}s] AI: {text}\n          {} expected an action containing `{must}`\n", if ok { "PASS" } else { "FAIL" });
                pressed.clear();
                asked += 1;
                if asked == COMMANDS.len() {
                    break;
                }
                ask(&to_ai, asked);
            }
            FromAi::Error(text) => {
                println!("ERROR: {text}");
                break;
            }
            _ => {}
        }
    }
    println!("{passed}/{} commands operated the app correctly", COMMANDS.len());
}

fn ask(to_ai: &std::sync::mpsc::Sender<ToAi>, i: usize) {
    println!("YOU: {}", COMMANDS[i].0);
    let _ = to_ai.send(ToAi::Said(COMMANDS[i].0.into()));
}

/// A stand-in app with a tracked, analysed swing clip open.
fn fake_app(action: Action) -> ActionResult {
    match action {
        Action::GetResults => ActionResult::Text(
            "Video: synthetic_swing.mp4 (360 frames, 12.0 s).\nPlayback: paused at frame 0 (0.00 s).\n\
             Marker colour: set. Scale: 400 px = 1000.0 mm.\nTracked frames: 360.\n\
             Settings: training window 5.0 s, sparsity λ 0.10, gravity x1.00, friction x1.00, speak replies on.\n\
             Neptune Mode: Off, visible part 100%, hidden pivot not set, reveal off.\n\
             Law: θ̈ = −19.59·sin θ − 0.30·θ̇ (g ≈ 9.79 m/s², L = 0.50 m)\n"
                .into(),
        ),
        Action::GetFrame => ActionResult::Frame(image::RgbImage::from_pixel(480, 270, image::Rgb([200, 200, 200]))),
        other => ActionResult::Text(format!("Done: {other:?}")),
    }
}
