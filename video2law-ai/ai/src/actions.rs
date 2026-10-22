//! The buttons the AI is allowed to press in the app ("tools" in AI jargon).
//!
//! Each tool has a name, a one-line description the model reads, and its parameters
//! as a JSON schema (Ollama's tool format). When the model decides to use one, we turn
//! its JSON arguments into an `Action` and ask the app to perform it.

use serde_json::{Value, json};

/// Something the app should do for the AI.
#[derive(Debug, Clone)]
pub enum Action {
    /// List the video files in clips/.
    ListClips,
    /// Open a video file (a name from ListClips, or a full path).
    OpenVideo { file: String },
    /// Give the AI the current video frame so it can look at it.
    GetFrame,
    /// Give the AI about `count` evenly spaced frames (for AI tracking).
    GetKeyframes { count: usize },
    /// The AI found an object at this box (frame pixels: x1, y1, x2, y2).
    /// `use_as`: "scale" (its width is `width_mm`) or "hidden_pivot" (Neptune Setup A).
    UseObject { bbox: [f64; 4], use_as: String, width_mm: Option<f64> },
    /// AI tracking: the object's box in some keyframes (frame index, box). The app
    /// follows the object between them by appearance, then builds the trajectory.
    UseAiTrack { boxes: Vec<(usize, [f64; 4])> },
    /// Classic tracking: follow the marker colour (set with UseObject use_as="marker").
    Track,
    /// Jump to a moment in the video (seconds) or a frame number.
    Seek { seconds: Option<f64>, frame: Option<usize> },
    /// STLSQ sparsity λ: how small a term can be before the law drops it.
    SetSparsity { lambda: f64 },
    /// Turn reading replies aloud on or off.
    SetSpeakReplies { on: bool },
    /// Run the physics engine on the tracked motion.
    DiscoverLaw,
    /// What-if sliders. `None` leaves a slider where it is.
    SetWhatIf { gravity_scale: Option<f64>, friction_scale: Option<f64> },
    /// How many seconds of the clip the law is learnt from (the ghost predicts the rest).
    SetTrainingWindow { seconds: f64 },
    /// Neptune Mode: "off", "coupled_pair" or "hidden_hand"; optional visible share of the frame.
    SetNeptune { mode: String, visible_fraction: Option<f64> },
    /// Show or hide the cropped-away part (the big reveal).
    Reveal { show: bool },
    /// Play or pause the video.
    Play { playing: bool },
    /// Everything the app currently knows: video, tracking, law, energy, ghost, Neptune.
    GetResults,
}

/// What the app answers.
pub enum ActionResult {
    /// A short text for the AI to read (what happened, or the current results).
    Text(String),
    /// The current frame, for the AI to look at.
    Frame(image::RgbImage),
    /// Keyframes: (frame index, image).
    Frames(Vec<(usize, image::RgbImage)>),
}

/// The tool list given to the model (Ollama / OpenAI function format).
pub fn tools() -> Value {
    let tool = |name: &str, description: &str, parameters: Value| {
        json!({"type": "function", "function": {"name": name, "description": description, "parameters": parameters}})
    };
    let none = json!({"type": "object", "properties": {}});
    json!([
        tool("list_clips", "List the video files available in the clips folder.", none.clone()),
        tool(
            "open_video",
            "Open a video file so it can be analysed.",
            json!({"type": "object", "properties": {"file": {"type": "string", "description": "File name from list_clips"}}, "required": ["file"]}),
        ),
        tool(
            "track_object",
            "AI tracking: find the named object (any object: ball, light, card, keys, pendulum bob...) in the video with the vision model and follow it through every frame. Use this before discover_law.",
            json!({"type": "object", "properties": {"description": {"type": "string", "description": "What to track, in a few words, e.g. 'the glowing ball'"}}, "required": ["description"]}),
        ),
        tool(
            "find_object",
            "Look at the current frame and find an object. use_as='scale' measures it to set the pixel-to-metre scale (give width_mm, its real width, e.g. 85.6 for an ID card); use_as='hidden_pivot' marks where the hidden object hangs from (Neptune Mode); use_as='marker' picks its colour for classic colour tracking.",
            json!({"type": "object", "properties": {
                "description": {"type": "string"},
                "use_as": {"type": "string", "enum": ["scale", "hidden_pivot", "marker"]},
                "width_mm": {"type": "number"}
            }, "required": ["description", "use_as"]}),
        ),
        tool("track_colour", "Classic tracking: follow the marker colour through every frame (fast). Needs find_object(use_as='marker') first. Prefer track_object unless the user asks for colour tracking.", none.clone()),
        tool(
            "seek",
            "Show a moment of the video: give seconds or a frame number.",
            json!({"type": "object", "properties": {"seconds": {"type": "number"}, "frame": {"type": "integer"}}}),
        ),
        tool(
            "set_sparsity",
            "Set the law-discovery sparsity lambda (0.01-1): bigger keeps fewer terms in the law. Then call discover_law again.",
            json!({"type": "object", "properties": {"lambda": {"type": "number"}}, "required": ["lambda"]}),
        ),
        tool("set_speak_replies", "Turn reading your replies aloud on or off.", json!({"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]})),
        tool("discover_law", "Run the physics engine on the tracked motion: law of motion, g, conserved energy, its leak, ghost prediction and (if on) Neptune Mode.", none.clone()),
        tool(
            "set_what_if",
            "Change the what-if sliders, then the ghost is re-predicted. gravity_scale: 1 = Earth, 0.166 = Moon. friction_scale: 1 = as filmed, 0 = no friction.",
            json!({"type": "object", "properties": {"gravity_scale": {"type": "number"}, "friction_scale": {"type": "number"}}}),
        ),
        tool(
            "set_training_window",
            "Set how many seconds at the start of the clip the law is learnt from. The ghost predicts everything after.",
            json!({"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}),
        ),
        tool(
            "set_neptune",
            "Neptune Mode: detect a hidden object. 'coupled_pair' = two ID cards on one cable, one hidden; 'hidden_hand' = keys held by a hidden hand; 'off'. visible_fraction crops the frame to its left part (e.g. 0.5).",
            json!({"type": "object", "properties": {
                "mode": {"type": "string", "enum": ["off", "coupled_pair", "hidden_hand"]},
                "visible_fraction": {"type": "number"}
            }, "required": ["mode"]}),
        ),
        tool("reveal", "Show (true) or hide (false) the cropped-away part of the video.", json!({"type": "object", "properties": {"show": {"type": "boolean"}}, "required": ["show"]})),
        tool("play", "Play (true) or pause (false) the video.", json!({"type": "object", "properties": {"playing": {"type": "boolean"}}, "required": ["playing"]})),
        tool("get_results", "Get the current state and all results (law, g, energy, leak, ghost, hidden objects). Use this before answering any question about the data.", none),
    ])
}

/// Turn a tool call (name + JSON arguments) into an `Action`.
/// `track_object` and `find_object` are not here: the brain handles them (they need the eyes).
pub fn parse(name: &str, args: &Value) -> Result<Action, String> {
    let num = |key: &str| args.get(key).and_then(Value::as_f64);
    let text = |key: &str| args.get(key).and_then(Value::as_str).map(str::to_string);
    let flag = |key: &str| args.get(key).and_then(Value::as_bool);
    Ok(match name {
        "list_clips" => Action::ListClips,
        "open_video" => Action::OpenVideo { file: text("file").ok_or("open_video needs 'file'")? },
        "discover_law" => Action::DiscoverLaw,
        "track_colour" => Action::Track,
        "seek" => Action::Seek { seconds: num("seconds"), frame: num("frame").map(|f| f.max(0.0) as usize) },
        "set_sparsity" => Action::SetSparsity { lambda: num("lambda").ok_or("needs 'lambda'")? },
        "set_speak_replies" => Action::SetSpeakReplies { on: flag("on").unwrap_or(true) },
        "set_what_if" => Action::SetWhatIf { gravity_scale: num("gravity_scale"), friction_scale: num("friction_scale") },
        "set_training_window" => Action::SetTrainingWindow { seconds: num("seconds").ok_or("needs 'seconds'")? },
        "set_neptune" => Action::SetNeptune {
            mode: text("mode").ok_or("set_neptune needs 'mode'")?,
            visible_fraction: num("visible_fraction"),
        },
        "reveal" => Action::Reveal { show: flag("show").unwrap_or(true) },
        "play" => Action::Play { playing: flag("playing").unwrap_or(true) },
        "get_results" => Action::GetResults,
        other => return Err(format!("unknown tool '{other}'")),
    })
}
