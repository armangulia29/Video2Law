//! The AI's brain, running locally through Ollama on this laptop's GPU.
//!
//! - Brain: Qwen3 4B reads the user's message, decides which app actions to use
//!   (tool calls), reads their results, and answers in plain words.
//! - Eyes: Qwen3-VL 2B looks at video frames. It finds the object to track in
//!   keyframes across the clip (AI tracking), the card to measure for the scale,
//!   and the hidden card's pivot.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use serde_json::{Value, json};

use crate::actions::{self, Action, ActionResult};
use crate::listen::Ears;
use crate::ollama::{self, BRAIN_MODEL, EYES_MODEL};
use crate::{FromAi, ToAi};

/// How the model was told to behave. Short and concrete: small models follow
/// short instructions best.
const SYSTEM_PROMPT: &str = "You are the assistant inside Video2Law, a desktop app that takes a \
video of ANY moving object (a ball, a pendulum, a card, keys, a glowing light...) and discovers its \
physics: the law of motion, g, a conserved energy and how fast it leaks, a 'ghost' prediction of the \
future, and Neptune Mode, which detects a hidden object coupled to the visible one. You are also a \
friendly physics teacher.\n\
You operate the whole app with tools, exactly like a user pressing its buttons: open videos, play, \
pause and jump in time, AI tracking (track_object) or colour tracking, set the scale, discover the law, \
the training window, sparsity, gravity and friction what-ifs (Moon = 0.166), Neptune Mode with its crop, \
hidden pivot and reveal, and reading aloud. 'Current app state' tells you every setting and what is \
already done: don't redo finished steps, and check it before answering. track_object can follow any \
object you can name: use the object described under 'What the video shows'. Workflow for a clip: \
open_video (if none is open), track_object, discover_law, get_results, then explain.\n\
Rules:\n\
- Every change in the app needs a tool call. Never say something was done (played, paused, \
moved, set, tracked, turned on or off) unless you called the tool for it in this turn and it succeeded. \
For 'pause and go to 3 seconds' call play(playing=false) and seek(seconds=3).\n\
- Examples are never limits: never refuse because an object isn't a card or keys.\n\
- Multi-part requests: do the tool steps first, then answer the rest (lesson plans, explanations, \
analogies) from your own knowledge.\n\
- Never invent numbers. Every number about the user's video comes from get_results.\n\
- For any question about the results (law, g, energy, ghost, hidden object), call get_results first; \
if it says no law yet, call discover_law, then get_results.\n\
- Keep answers short and clear for students (the answer may be read aloud), unless the user asks \
for a plan or a longer explanation.\n\
- Write plain text: no Markdown symbols (no **, ##, ---) and no emoji; use simple numbered lines for lists.\n\
- If a step fails, say what went wrong and what the user can do.\n\
- When the user asks you to do something, call the tool right away; never describe what you are about to do.";

/// Maximum tool rounds per user message, so a confused model can't loop forever.
const MAX_ROUNDS: usize = 4;
/// AI tracking: how many keyframes the eyes look at (≈1 s each on a GTX 1650).
const KEYFRAMES: usize = 30;

/// The AI thread: start Ollama, check the models, then answer messages until the app closes.
pub fn run(from_app: Receiver<ToAi>, to_app: Sender<FromAi>, models: PathBuf) {
    let _ = to_app.send(FromAi::Loading("Starting the local AI (Ollama)...".into()));
    if let Err(e) = ollama::ensure_running() {
        let _ = to_app.send(FromAi::Error(e));
        return;
    }
    for model in [BRAIN_MODEL, EYES_MODEL] {
        if !ollama::has_model(model) {
            let _ = to_app.send(FromAi::Error(format!(
                "Model {model} isn't downloaded. With OLLAMA_MODELS set to runtime\\models\\ollama, run: runtime\\ollama\\ollama.exe pull {model}"
            )));
            return;
        }
    }
    // Warm the brain up so the first real question is quick.
    let _ = ollama::chat(BRAIN_MODEL, &[json!({"role": "user", "content": "hi"})], None, 1);

    // Speech-to-text is optional: without it, typing still works.
    let mut ears = Ears::load(&models.join("whisper-base-en"))
        .map_err(|e| {
            let _ = to_app.send(FromAi::Loading(format!("Voice input off: {e}")));
        })
        .ok();
    let _ = to_app.send(FromAi::Ready);

    let mut brain = Brain { history: Vec::new(), scene: None, to_app: to_app.clone() };
    for message in from_app {
        let text = match message {
            ToAi::Said(text) => text,
            ToAi::Spoke(samples) => match ears.as_mut().map(|e| e.transcribe(&samples)) {
                Some(Ok(text)) if !text.trim().is_empty() => {
                    let _ = to_app.send(FromAi::Heard(text.clone()));
                    text
                }
                Some(Ok(_)) => continue, // silence
                Some(Err(e)) => {
                    let _ = to_app.send(FromAi::Error(format!("Didn't catch that: {e}")));
                    continue;
                }
                None => {
                    let _ = to_app.send(FromAi::Error("Voice input isn't available; please type.".into()));
                    continue;
                }
            },
        };
        let _ = to_app.send(FromAi::Thinking);
        let reply = brain.answer(&text).unwrap_or_else(|e| format!("Sorry, something went wrong: {e}"));
        let _ = to_app.send(FromAi::Reply(reply));
    }
}

struct Brain {
    history: Vec<Value>, // earlier user/assistant turns (kept short)
    /// What the eyes saw in the open video: (video line from the app state, description).
    /// Refreshed only when a different video is opened (it takes the eyes ~5 s).
    scene: Option<(String, String)>,
    to_app: Sender<FromAi>,
}

impl Brain {
    /// What the brain needs to know before every answer: the app's live state
    /// (video, tracking, results) and, once per video, what the video shows.
    /// Without this a small model has no idea a video is even open.
    fn context(&mut self) -> String {
        let state = match self.ask_app(Action::GetResults) {
            Ok(ActionResult::Text(text)) => text,
            _ => "App state unavailable.".into(),
        };
        let video_line = state.lines().next().unwrap_or("").to_string();
        if video_line.starts_with("Video:") && self.scene.as_ref().map(|s| &s.0) != Some(&video_line) {
            let _ = self.to_app.send(FromAi::Loading("Looking at the video...".into()));
            // Remember the answer even if the eyes saw nothing clear, so we don't
            // swap models and ask again on every turn for the same video.
            let description = match self.ask_app(Action::GetFrame) {
                Ok(ActionResult::Frame(frame)) => describe(&frame).unwrap_or_else(|_| "no clear object seen".into()),
                _ => "unknown".into(),
            };
            self.scene = Some((video_line.clone(), description));
        }
        let shows = match &self.scene {
            Some((line, description)) if *line == video_line => description.clone(),
            _ => "unknown (no video open, or not looked at yet)".into(),
        };
        format!("Current app state:\n{state}\nWhat the video shows: {shows}")
    }

    /// One user message → plan actions → do them → answer from their real results.
    ///
    /// A small model treats optional tool calls as optional: it would sometimes just
    /// *say* "the video is paused" without pausing it. So the decision is made
    /// mandatory and structured: first the brain must fill in a JSON plan
    /// ({"actions": [...]}, enforced by Ollama), the app performs those actions, and
    /// only then does the brain write its reply, from the actual results.
    fn answer(&mut self, user_text: &str) -> Result<String, String> {
        let context = self.context();
        let mut base = vec![
            json!({"role": "system", "content": SYSTEM_PROMPT}),
            json!({"role": "system", "content": context}),
        ];
        base.extend(self.history.iter().cloned());
        base.push(json!({"role": "user", "content": user_text}));

        // 1. Plan → act, up to MAX_ROUNDS times (multi-step jobs: track, then discover, ...).
        // The planner sees only the live app state and the CURRENT message, not the
        // earlier chat: otherwise a small model copies the previous turn's actions.
        let planner_base = vec![
            json!({"role": "system", "content": SYSTEM_PROMPT}),
            json!({"role": "system", "content": context.clone()}),
            json!({"role": "user", "content": user_text}),
        ];
        let mut done: Vec<String> = Vec::new(); // "action(args) → result" lines
        for _ in 0..MAX_ROUNDS {
            let mut messages = planner_base.clone();
            if !done.is_empty() {
                messages.push(json!({"role": "system", "content": format!("Already done this turn:\n{}", done.join("\n"))}));
            }
            messages.push(json!({"role": "system", "content": plan_instructions()}));
            let reply = ollama::chat_with_format(BRAIN_MODEL, &messages, None, Some(&plan_schema()), 400)?;
            let plan: Value = serde_json::from_str(reply["content"].as_str().unwrap_or("{}")).unwrap_or_default();
            let steps = plan["actions"].as_array().cloned().unwrap_or_default();
            // Small models tend to re-plan what they just did: skip exact repeats, and stop
            // planning once a round brings nothing new.
            let mut anything_new = false;
            for step in &steps {
                let name = step["name"].as_str().unwrap_or("");
                let args = step.get("arguments").cloned().unwrap_or(json!({}));
                // Compare the parsed action, so `0` and `0.0` count as the same repeat.
                let call = actions::parse(name, &args).map_or_else(|_| format!("{name}({args})"), |a| format!("{a:?}"));
                if done.iter().any(|d| d.starts_with(&call)) {
                    continue;
                }
                anything_new = true;
                let result = self.use_tool(name, &args);
                done.push(format!("{call} → {result}"));
            }
            if !anything_new {
                break;
            }
        }

        // 2. Answer, describing only what really happened.
        let mut messages = base;
        messages.push(json!({"role": "system", "content": if done.is_empty() {
            "No app actions were taken for this message. Answer the user; do not claim that anything in the app changed.".to_string()
        } else {
            format!("Actions taken for this message and their results:\n{}\nNow answer the user. Describe only what these results show.", done.join("\n"))
        }}));
        let reply = ollama::chat(BRAIN_MODEL, &messages, None, 1500)?; // room for full lesson plans
        let text = clean(reply["content"].as_str().unwrap_or(""));
        self.remember(user_text, &text);
        Ok(text)
    }

    /// Keep only the last few exchanges: small models get confused by long histories.
    fn remember(&mut self, user: &str, assistant: &str) {
        self.history.push(json!({"role": "user", "content": user}));
        self.history.push(json!({"role": "assistant", "content": assistant}));
        let excess = self.history.len().saturating_sub(8);
        self.history.drain(..excess);
    }

    /// Run one tool and describe the outcome in words for the model.
    fn use_tool(&self, name: &str, args: &Value) -> String {
        let outcome = match name {
            "track_object" => self.track_object(args),
            "find_object" => self.find_object(args),
            _ => actions::parse(name, args).and_then(|action| match self.ask_app(action)? {
                ActionResult::Text(text) => Ok(text),
                _ => Ok("ok".into()),
            }),
        };
        outcome.unwrap_or_else(|e| format!("{name} failed: {e}"))
    }

    /// Ask the app to do something and wait for its answer.
    /// Tracking and analysis can take a while, so wait up to 5 minutes.
    fn ask_app(&self, action: Action) -> Result<ActionResult, String> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.to_app.send(FromAi::Do(action, reply_tx)).map_err(|_| "the app closed")?;
        reply_rx.recv_timeout(Duration::from_secs(300)).map_err(|_| "the app didn't answer".to_string())
    }

    /// AI tracking: the eyes find the object in keyframes spread over the clip; the
    /// app then follows it between keyframes by its appearance.
    fn track_object(&self, args: &Value) -> Result<String, String> {
        let description = args.get("description").and_then(Value::as_str).ok_or("needs 'description'")?;
        let ActionResult::Frames(keyframes) = self.ask_app(Action::GetKeyframes { count: KEYFRAMES })? else {
            return Err("no video open".into());
        };
        let total = keyframes.len();
        let mut boxes = Vec::new();
        for (i, (index, frame)) in keyframes.iter().enumerate() {
            let _ = self.to_app.send(FromAi::Loading(format!("AI tracking: looking at keyframe {}/{total}", i + 1)));
            if let Ok(bbox) = locate(frame, description) {
                boxes.push((*index, bbox));
            }
        }
        if boxes.len() < 2 {
            return Err(format!("the vision model couldn't find {description} (found in {}/{total} keyframes)", boxes.len()));
        }
        let found = boxes.len();
        match self.ask_app(Action::UseAiTrack { boxes })? {
            ActionResult::Text(text) => Ok(format!("AI found {description} in {found}/{total} keyframes. {text}")),
            _ => Ok(format!("AI found {description} in {found}/{total} keyframes.")),
        }
    }

    /// Look at the current frame, find the described object, and tell the app to use it.
    fn find_object(&self, args: &Value) -> Result<String, String> {
        let description = args.get("description").and_then(Value::as_str).ok_or("needs 'description'")?;
        let use_as = args.get("use_as").and_then(Value::as_str).unwrap_or("scale").to_string();
        let width_mm = args.get("width_mm").and_then(Value::as_f64);
        let ActionResult::Frame(frame) = self.ask_app(Action::GetFrame)? else {
            return Err("no video open".into());
        };
        let bbox = locate(&frame, description)?;
        match self.ask_app(Action::UseObject { bbox, use_as, width_mm })? {
            ActionResult::Text(text) => Ok(text),
            _ => Ok(format!("found {description}")),
        }
    }
}

/// Ask the eyes what the main moving object in `frame` is, in one short phrase.
fn describe(frame: &image::RgbImage) -> Result<String, String> {
    let prompt = "In one short sentence, name the main object in this image that could be moving \
(for example 'a glowing ball', 'an ID card on a lanyard', 'keys on a string') and where it is.";
    let message = json!({"role": "user", "content": prompt, "images": [ollama::image_b64(frame)]});
    let reply = ollama::chat(EYES_MODEL, &[message], None, 700)?;
    let text = clean(reply["content"].as_str().unwrap_or(""));
    if text.is_empty() { Err("no description".into()) } else { Ok(text) }
}

/// Ask the eyes for a bounding box around `description` in `frame`.
/// Returns (x1, y1, x2, y2) in frame pixels.
pub fn locate(frame: &image::RgbImage, description: &str) -> Result<[f64; 4], String> {
    let prompt = format!(
        "Locate {description} in the image. Reply with only JSON: {{\"bbox_2d\": [x1, y1, x2, y2]}}."
    );
    let message = json!({"role": "user", "content": prompt, "images": [ollama::image_b64(frame)]});
    // The eyes model thinks first (~300 tokens) before answering, so give it room.
    let reply = ollama::chat(EYES_MODEL, &[message], None, 700)?;
    let text = reply["content"].as_str().unwrap_or("");
    parse_bbox(text, frame.width() as f64, frame.height() as f64)
        .ok_or_else(|| format!("couldn't see {description} (model said: {text})"))
}

/// Qwen3-VL gives boxes on a 0–1000 grid regardless of the image size
/// (Qwen2.5-VL used real pixels). Convert to frame pixels.
fn parse_bbox(text: &str, width: f64, height: f64) -> Option<[f64; 4]> {
    let start = text.find('[')?;
    let end = start + text[start..].find(']')?;
    let numbers: Vec<f64> = text[start + 1..end].split(',').filter_map(|n| n.trim().parse().ok()).collect();
    let [x1, y1, x2, y2] = numbers[..] else { return None };
    let box_px = [x1 / 1000.0 * width, y1 / 1000.0 * height, x2 / 1000.0 * width, y2 / 1000.0 * height];
    (box_px[2] > box_px[0] && box_px[3] > box_px[1]).then_some(box_px)
}

/// What the planning step is told: the app's buttons (tools) and how to choose them.
fn plan_instructions() -> String {
    let tools = actions::tools();
    let list: Vec<String> = tools
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| {
            let f = &t["function"];
            format!("- {}: {} Arguments: {}", f["name"].as_str().unwrap_or(""), f["description"].as_str().unwrap_or(""), f["parameters"]["properties"])
        })
        .collect();
    format!(
        "Decide which app actions to perform NOW for the user's latest message. Available actions:\n{}\n\
         Reply with JSON {{\"actions\": [{{\"name\": ..., \"arguments\": {{...}}}}, ...]}} in the order to run them.\n\
         - Include every change the user asked for. Examples: pause = play with playing=false; go to 3 s = seek \
         seconds=3; Moon = set_what_if gravity_scale=0.166; no friction = set_what_if friction_scale=0; \
         stop speaking = set_speak_replies on=false.\n\
         - For questions about results, use get_results (and discover_law first if there is no law yet).\n\
         - Use {{\"actions\": []}} if nothing (more) needs doing: a pure question, or everything asked is already done.\n\
         - Never repeat an action listed under 'Already done this turn'.",
        list.join("\n")
    )
}

/// The JSON shape the plan must have (Ollama enforces it).
fn plan_schema() -> Value {
    let names: Vec<Value> = actions::tools()
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t["function"]["name"].as_str().map(|n| json!(n)))
        .collect();
    json!({
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": names}, "arguments": {"type": "object"}},
                    "required": ["name", "arguments"]
                }
            }
        },
        "required": ["actions"]
    })
}

/// Remove any <think>…</think> reasoning a model might add before its answer.
fn clean(text: &str) -> String {
    match text.find("</think>") {
        Some(i) => text[i + "</think>".len()..].trim().to_string(),
        None => text.trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_on_the_1000_grid_becomes_pixels() {
        let b = parse_bbox("```json\n{\"bbox_2d\": [100, 200, 300, 400]}\n```", 480.0, 270.0).unwrap();
        assert_eq!(b, [48.0, 54.0, 144.0, 108.0]);
        assert!(parse_bbox("I can't see it", 480.0, 270.0).is_none());
    }
}
