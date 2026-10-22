//! Video2Law's built-in AI assistant. Runs fully on this laptop, nothing goes online.
//!
//! - `brain`: Qwen3 4B (chat, app control) + Qwen3-VL 2B (eyes, AI tracking) via Ollama,
//!   running locally on the GPU; it drives the app by calling `actions`.
//! - `listen`: Whisper speech-to-text, so you can talk to it.
//! - `voice`: microphone recording and Windows' built-in speaking voice.
//!
//! The AI runs on its own thread (it is slow and must never freeze the window).
//! The app and the AI talk through two channels: `ToAi` messages go in,
//! `FromAi` messages come out. When the AI wants to press a button in the app it
//! sends `FromAi::Do(action, reply)`; the app does it and answers on `reply`.
//!
//! All physics numbers the AI quotes come from the engine via those actions,
//! never from the model's imagination.

pub mod actions;
pub mod brain;
pub mod ollama;
pub mod listen;
pub mod voice;

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};

pub use actions::{Action, ActionResult};

/// The `runtime/` folder holding Ollama and the models (GBs; never inside Drive or git).
///
/// Looked up so the same build works from the source tree, from `dist/`, or copied
/// elsewhere: `V2L_RUNTIME` if set, else the first `runtime/` folder found next to
/// the .exe or the current folder, or in any folder above them.
pub fn runtime_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("V2L_RUNTIME") {
        return PathBuf::from(dir);
    }
    let starts = [std::env::current_exe().ok(), std::env::current_dir().ok()];
    for start in starts.into_iter().flatten() {
        for folder in start.ancestors() {
            let candidate = folder.join("runtime");
            if candidate.join("ollama").join("ollama.exe").exists() {
                return candidate;
            }
        }
    }
    PathBuf::from("runtime") // not found: errors later will name this path
}

/// Where the model files live: `runtime/models`.
pub fn models_dir() -> PathBuf {
    runtime_dir().join("models")
}

/// Messages from the app to the AI.
pub enum ToAi {
    /// The user typed something in the chat box.
    Said(String),
    /// The user spoke: 16 kHz mono samples from the microphone.
    Spoke(Vec<f32>),
}

/// Messages from the AI to the app.
pub enum FromAi {
    /// Progress while the models load (takes a minute or more).
    Loading(String),
    /// Models loaded; the AI is listening.
    Ready,
    /// What speech-to-text understood, shown in the chat as the user's line.
    Heard(String),
    /// The AI is working on an answer (show a spinner).
    Thinking,
    /// The AI's answer, to show in the chat and speak aloud.
    Reply(String),
    /// The AI wants the app to do something. Answer on the sender when it's done.
    Do(Action, Sender<ActionResult>),
    /// Something went wrong (e.g. models missing). Shown in the chat.
    Error(String),
}

/// Start the AI on a background thread. Returns the two ends the app keeps.
pub fn start() -> (Sender<ToAi>, Receiver<FromAi>) {
    let (to_ai, from_app) = mpsc::channel();
    let (to_app, from_ai) = mpsc::channel();
    std::thread::spawn(move || brain::run(from_app, to_app, models_dir()));
    (to_ai, from_ai)
}
