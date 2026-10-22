//! Talking to Ollama, the local AI runtime that runs our models on this laptop's GPU.
//!
//! Ollama is a separate program (like ffmpeg): we start it if it isn't running and
//! send it JSON over http://127.0.0.1:11434. Nothing leaves the laptop.

use std::io::Cursor;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine;
use serde_json::{Value, json};

/// Where Ollama listens. Local only.
const URL: &str = "http://127.0.0.1:11434";
/// The portable Ollama in runtime/ollama, and where its models live (runtime/models/ollama).
fn ollama_exe() -> std::path::PathBuf {
    crate::runtime_dir().join("ollama").join("ollama.exe")
}
fn ollama_models() -> std::path::PathBuf {
    crate::models_dir().join("ollama")
}

/// The brain (chat + app control) and the eyes (finding objects in frames).
pub const BRAIN_MODEL: &str = "qwen3:4b-instruct"; // the non-thinking edition: answers and calls tools directly
pub const EYES_MODEL: &str = "qwen3-vl:2b";

/// Make sure Ollama is running; start it if not.
pub fn ensure_running() -> Result<(), String> {
    if ureq::get(&format!("{URL}/api/version")).call().is_ok() {
        return Ok(());
    }
    let exe = ollama_exe();
    Command::new(&exe)
        .arg("serve")
        .env("OLLAMA_MODELS", ollama_models())
        .env("OLLAMA_HOST", "127.0.0.1:11434")
        // Store the conversation memory (KV cache) at 8-bit instead of 16-bit: half the
        // GPU memory, so more of the model fits on the 4 GB GTX 1650.
        .env("OLLAMA_FLASH_ATTENTION", "1")
        .env("OLLAMA_KV_CACHE_TYPE", "q8_0")
        // Only one model on the 4 GB GPU at a time: brain and eyes swap cleanly.
        .env("OLLAMA_MAX_LOADED_MODELS", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("couldn't start Ollama ({}): {e}", exe.display()))?;
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(300));
        if ureq::get(&format!("{URL}/api/version")).call().is_ok() {
            return Ok(());
        }
    }
    Err("Ollama didn't start within 20 s".into())
}

/// Is this model downloaded? (`ollama pull <name>` downloads it.)
pub fn has_model(name: &str) -> bool {
    ureq::get(&format!("{URL}/api/tags"))
        .call()
        .ok()
        .and_then(|r| r.into_json::<Value>().ok())
        .and_then(|v| v["models"].as_array().cloned())
        .is_some_and(|models| models.iter().any(|m| m["name"].as_str() == Some(name)))
}

/// One chat request. `messages` and `tools` are Ollama's JSON format.
/// Returns the assistant message (with `content` and maybe `tool_calls`).
pub fn chat(model: &str, messages: &[Value], tools: Option<&Value>, max_tokens: u32) -> Result<Value, String> {
    chat_with_format(model, messages, tools, None, max_tokens)
}

/// Like `chat`, but `format` (a JSON schema) forces the reply to be JSON of that shape.
pub fn chat_with_format(
    model: &str,
    messages: &[Value],
    tools: Option<&Value>,
    format: Option<&Value>,
    max_tokens: u32,
) -> Result<Value, String> {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "think": false,          // answer directly, no long hidden reasoning
        "keep_alive": "30m",     // stay loaded on the GPU between questions
        "options": { "temperature": 0.2, "num_ctx": 4096, "num_predict": max_tokens }
    });
    if let Some(tools) = tools {
        body["tools"] = tools.clone();
    }
    if let Some(format) = format {
        body["format"] = format.clone();
    }
    let response = ureq::post(&format!("{URL}/api/chat"))
        .timeout(Duration::from_secs(300))
        .send_json(body)
        .map_err(|e| match e {
            ureq::Error::Status(code, r) => format!("Ollama error {code}: {}", r.into_string().unwrap_or_default()),
            other => format!("can't reach Ollama: {other}"),
        })?;
    let reply: Value = response.into_json().map_err(|e| e.to_string())?;
    Ok(reply["message"].clone())
}

/// A frame as base64 JPEG, the way Ollama takes images.
pub fn image_b64(frame: &image::RgbImage) -> String {
    let mut jpeg = Cursor::new(Vec::new());
    let _ = image::DynamicImage::ImageRgb8(frame.clone()).write_to(&mut jpeg, image::ImageFormat::Jpeg);
    base64::engine::general_purpose::STANDARD.encode(jpeg.into_inner())
}
