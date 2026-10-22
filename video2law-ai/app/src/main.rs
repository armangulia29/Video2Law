//! Video2Law desktop app: everything on screen. The maths lives in the `engine` crate.

mod assistant;
mod equation;
mod overlay;
mod plots;
mod source;
mod tracker;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui::{
    self, Color32, ColorImage, ComboBox, RichText, Sense, Slider, TextureHandle, TextureOptions,
};
use engine::{NeptuneMode, Options, Report, Trajectory};
use rfd::FileDialog;

use source::{probe_video, start_decoding, VideoFrame, VideoMeta};
use tracker::{ColorRgb, Tracker};

fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Video2Law")
            .with_inner_size([1360.0, 840.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Video2Law",
        native_options,
        Box::new(|_cc| Ok(Box::new(Video2LawApp::default()))),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClickMode {
    PickMarker,
    ScalePoint1,
    ScalePoint2,
    HiddenPivot, // Neptune Setup A: where the hidden card hangs from
}

/// Results coming back from the background worker thread.
enum WorkDone {
    Tracked(Tracker, Trajectory),
    Analysed(Report),
}

/// What an AI request is waiting for before the app can answer it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Waiting {
    VideoLoaded,
    Tracked,
    Analysed,
}

struct Video2LawApp {
    // Video state
    meta: Option<VideoMeta>,
    frames: Vec<Arc<VideoFrame>>,
    current_frame: usize,
    is_playing: bool,
    last_frame_tick: Instant,
    rx_frames: Option<Receiver<VideoFrame>>,
    cancel_decoding: Option<Arc<AtomicBool>>,
    video_texture: Option<TextureHandle>,
    video_texture_idx: Option<usize>,

    // Tracking state
    tracker: Tracker,
    click_mode: ClickMode,
    scale_dist_mm: f64,

    // Engine contract
    trajectory: Trajectory,
    options: Options,
    report: Report,

    // Background work (CLAUDE.md: heavy work runs on a thread, results come back over mpsc)
    worker: Option<Receiver<WorkDone>>,
    analysed_with: Option<Options>, // the options behind `report`, to re-run when a slider moves

    // Neptune Mode
    hidden_pivot: Option<[f64; 2]>, // pixel position of the hidden card's pivot (Setup A)
    reveal: bool,                   // show the cropped-away part of the frame

    // Built-in AI assistant (chat + voice). It drives the app through `perform_ai_action`.
    assistant: assistant::Assistant,
    ai_waiting: Option<(Waiting, Sender<ai::ActionResult>)>, // an AI request finishing later
    video_name: String,

    status_message: String,
}

impl Default for Video2LawApp {
    fn default() -> Self {
        Self {
            meta: None,
            frames: Vec::new(),
            current_frame: 0,
            is_playing: false,
            last_frame_tick: Instant::now(),
            rx_frames: None,
            cancel_decoding: None,
            video_texture: None,
            video_texture_idx: None,

            tracker: Tracker::default(),
            click_mode: ClickMode::PickMarker,
            scale_dist_mm: 85.6, // ID card standard width (85.6 mm)

            trajectory: Trajectory::default(),
            options: Options::default(),
            report: Report::default(),

            worker: None,
            analysed_with: None,

            hidden_pivot: None,
            reveal: false,

            assistant: assistant::Assistant::default(),
            ai_waiting: None,
            video_name: String::new(),

            status_message: "Ready. Click 'Open Video' to load a swinging video clip.".to_string(),
        }
    }
}

/// The marker colour inside a box the AI found: the average of its most colourful
/// pixels (a sticker is colourful; the wall and the white card around it aren't).
fn marker_colour(frame: &VideoFrame, bbox: [f64; 4]) -> ColorRgb {
    let clamp_x = |v: f64| (v.max(0.0) as usize).min(frame.width - 1);
    let clamp_y = |v: f64| (v.max(0.0) as usize).min(frame.height - 1);
    let mut pixels: Vec<(f64, [u8; 3])> = Vec::new();
    for y in clamp_y(bbox[1])..=clamp_y(bbox[3]) {
        for x in clamp_x(bbox[0])..=clamp_x(bbox[2]) {
            let i = (y * frame.width + x) * 3;
            let p = [frame.rgb[i], frame.rgb[i + 1], frame.rgb[i + 2]];
            let (max, min) = (*p.iter().max().unwrap() as f64, *p.iter().min().unwrap() as f64);
            pixels.push((max - min, p)); // "colourfulness": how far from grey
        }
    }
    pixels.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top = &pixels[..(pixels.len() / 4).max(1)];
    let avg = |k: usize| (top.iter().map(|(_, p)| p[k] as f64).sum::<f64>() / top.len() as f64) as u8;
    ColorRgb::new(avg(0), avg(1), avg(2))
}

/// Did any setting that changes the analysis move?
fn same_options(a: &Options, b: &Options) -> bool {
    (a.threshold, a.train_until, a.gravity_scale, a.damping_scale, a.neptune)
        == (b.threshold, b.train_until, b.gravity_scale, b.damping_scale, b.neptune)
}

impl Video2LawApp {
    fn load_video_file(&mut self, path: PathBuf) {
        if let Some(cancel) = &self.cancel_decoding {
            cancel.store(true, Ordering::Relaxed);
        }

        self.frames.clear();
        self.current_frame = 0;
        self.is_playing = false;
        self.video_texture = None;
        self.video_texture_idx = None;
        self.trajectory = Trajectory::default();
        self.tracker.tracked_points.clear();
        self.report = Report::default();
        self.analysed_with = None;

        match probe_video(&path) {
            Ok(meta) => {
                self.meta = Some(meta);
                self.video_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                self.status_message = format!(
                    "Loading '{}' ({}x{} @ {:.2} fps)...",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    meta.width,
                    meta.height,
                    meta.fps
                );

                let (rx, cancel) = start_decoding(path, meta);
                self.rx_frames = Some(rx);
                self.cancel_decoding = Some(cancel);
            }
            Err(e) => {
                self.status_message = format!("Error probing video: {}", e);
            }
        }
    }

    fn update_decoding(&mut self) {
        let Some(rx) = &self.rx_frames else { return };
        // Take at most 60 frames per screen refresh so the window stays smooth.
        for _ in 0..60 {
            match rx.try_recv() {
                Ok(frame) => self.frames.push(Arc::new(frame)),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.rx_frames = None;
                    self.status_message = format!(
                        "Loaded {} frames. Pick the marker colour, set the scale, then 'Track Motion'.",
                        self.frames.len()
                    );
                    self.ai_done(Waiting::VideoLoaded);
                    return;
                }
            }
        }
    }

    fn is_busy(&self) -> bool {
        self.worker.is_some()
    }

    /// Track the sticker in every frame, on a background thread.
    fn run_tracking(&mut self) {
        if self.tracker.target_color.is_none() {
            self.status_message = "Click on the marker sticker in the video first to select its color.".to_string();
            return;
        }
        if self.frames.is_empty() || self.rx_frames.is_some() {
            self.status_message = "Wait until the whole video has loaded.".to_string();
            return;
        }
        if self.is_busy() {
            return;
        }

        self.status_message = format!("Tracking dot across {} frames...", self.frames.len());
        let mut tracker = self.tracker.clone();
        let frames = self.frames.clone(); // cheap: only the Arc pointers are copied
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let traj = tracker.track_frames(&frames);
            let _ = tx.send(WorkDone::Tracked(tracker, traj));
        });
        self.worker = Some(rx);
    }

    /// Run the engine on the tracked trajectory, on a background thread.
    fn run_analysis(&mut self) {
        if self.trajectory.t.is_empty() {
            self.status_message = "Track motion first before running law discovery.".to_string();
            return;
        }
        if self.is_busy() {
            return;
        }
        self.status_message = "Discovering the law...".to_string();
        let traj = self.trajectory.clone();
        let options = self.options.clone();
        self.analysed_with = Some(options.clone());
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(WorkDone::Analysed(engine::analyse(&traj, &options)));
        });
        self.worker = Some(rx);
    }

    /// Collect a finished background job, if any.
    fn update_worker(&mut self) {
        let Some(rx) = &self.worker else { return };
        match rx.try_recv() {
            Ok(WorkDone::Tracked(tracker, traj)) => {
                let missed = tracker.tracked_points.iter().filter(|p| p.is_none()).count();
                self.tracker = tracker;
                self.trajectory = traj;
                self.worker = None;
                self.status_message = format!(
                    "Tracked {} frames ({} missed and filled in). Click 'Discover Law'.",
                    self.trajectory.t.len(),
                    missed
                );
                // Re-tracking (e.g. after moving the crop) refreshes the results too.
                if self.analysed_with.is_some() {
                    self.run_analysis();
                }
                self.ai_done(Waiting::Tracked);
                if matches!(self.ai_waiting, Some((Waiting::Analysed, _))) && !self.is_busy() {
                    self.run_analysis();
                }
            }
            Ok(WorkDone::Analysed(report)) => {
                self.report = report;
                self.worker = None;
                self.status_message = format!("Law discovered: {}", self.report.law_plain);
                self.ai_done(Waiting::Analysed);
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.worker = None;
                self.status_message = "Background job failed.".to_string();
            }
        }
    }

    // ── Built-in AI assistant ────────────────────────────────────────────────────

    /// If an AI request was waiting for `what`, answer it now.
    fn ai_done(&mut self, what: Waiting) {
        if matches!(self.ai_waiting, Some((w, _)) if w == what) {
            let (_, reply) = self.ai_waiting.take().unwrap();
            let text = match what {
                Waiting::Analysed => self.results_summary(),
                _ => self.status_message.clone(),
            };
            let _ = reply.send(ai::ActionResult::Text(text));
        }
    }

    /// Answer now, or remember the request if the job is still running.
    /// (If the job couldn't even start, `status_message` says why.)
    fn ai_reply_when(&mut self, what: Waiting, started: bool, reply: Sender<ai::ActionResult>) {
        if started {
            self.ai_waiting = Some((what, reply));
        } else {
            let _ = reply.send(ai::ActionResult::Text(self.status_message.clone()));
        }
    }

    /// Do what the AI asked, like a user pressing the buttons.
    fn perform_ai_action(&mut self, action: ai::Action, reply: Sender<ai::ActionResult>) {
        use ai::{Action, ActionResult::Text};
        self.assistant.note(format!("→ {action:?}"));
        match action {
            Action::ListClips => {
                let names: Vec<String> = std::fs::read_dir("clips")
                    .map(|dir| {
                        dir.filter_map(|e| e.ok()?.file_name().into_string().ok())
                            .filter(|n| [".mp4", ".mov", ".avi", ".mkv"].iter().any(|ext| n.to_lowercase().ends_with(ext)))
                            .collect()
                    })
                    .unwrap_or_default();
                let _ = reply.send(Text(format!("Clips: {}", names.join(", "))));
            }
            Action::OpenVideo { file } => {
                let path = if std::path::Path::new(&file).exists() { PathBuf::from(&file) } else { PathBuf::from("clips").join(&file) };
                if !path.exists() {
                    let _ = reply.send(Text(format!("No file '{file}' in clips/.")));
                    return;
                }
                self.load_video_file(path);
                let started = self.rx_frames.is_some();
                self.ai_reply_when(Waiting::VideoLoaded, started, reply);
            }
            Action::GetFrame => match self.frames.get(self.current_frame.min(self.frames.len().saturating_sub(1))) {
                Some(f) => {
                    let img = image::RgbImage::from_raw(f.width as u32, f.height as u32, f.rgb.clone());
                    let _ = reply.send(img.map_or(Text("frame unreadable".into()), ai::ActionResult::Frame));
                }
                None => {
                    let _ = reply.send(Text("No video is open.".into()));
                }
            },
            Action::UseObject { bbox, use_as, width_mm } => {
                let text = self.use_found_object(bbox, &use_as, width_mm);
                let _ = reply.send(Text(text));
            }
            Action::Track => {
                self.run_tracking();
                let started = self.is_busy();
                self.ai_reply_when(Waiting::Tracked, started, reply);
            }
            Action::Seek { seconds, frame } => {
                let last = self.frames.len().saturating_sub(1);
                let target = match (frame, seconds) {
                    (Some(f), _) => f.min(last),
                    (None, Some(s)) => self.frames.iter().position(|f| f.time >= s).unwrap_or(last),
                    (None, None) => self.current_frame,
                };
                self.current_frame = target;
                self.is_playing = false;
                let t = self.frames.get(target).map_or(0.0, |f| f.time);
                let _ = reply.send(Text(format!("Showing frame {target} ({t:.2} s).")));
            }
            Action::SetSparsity { lambda } => {
                self.options.threshold = lambda.clamp(0.01, 1.0);
                self.run_analysis();
                let started = self.is_busy();
                self.ai_reply_when(Waiting::Analysed, started, reply);
            }
            Action::SetSpeakReplies { on } => {
                self.assistant.set_speaks(on);
                let _ = reply.send(Text(format!("Speaking replies is now {}.", if on { "on" } else { "off" })));
            }
            Action::GetKeyframes { count } => {
                // Evenly spaced frames across the clip, for the AI's eyes.
                let n = self.frames.len();
                let picks: Vec<(usize, image::RgbImage)> = (0..count.min(n))
                    .map(|k| k * (n - 1) / count.max(2).saturating_sub(1).max(1))
                    .filter(|&i| i < n)
                    .filter_map(|i| {
                        let f = &self.frames[i];
                        image::RgbImage::from_raw(f.width as u32, f.height as u32, f.rgb.clone()).map(|img| (i, img))
                    })
                    .collect();
                let _ = reply.send(if picks.is_empty() { Text("No video is open.".into()) } else { ai::ActionResult::Frames(picks) });
            }
            Action::UseAiTrack { boxes } => {
                // AI tracking: follow the object between the AI's keyframe sightings.
                if self.is_busy() || self.frames.is_empty() {
                    let _ = reply.send(Text("Busy or no video open.".into()));
                    return;
                }
                self.status_message = format!("AI tracking: following the object between {} AI sightings...", boxes.len());
                let mut tracker = self.tracker.clone();
                let frames = self.frames.clone();
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || {
                    let traj = tracker.track_with_ai(&frames, &boxes);
                    let _ = tx.send(WorkDone::Tracked(tracker, traj));
                });
                self.worker = Some(rx);
                self.ai_reply_when(Waiting::Tracked, true, reply);
            }
            Action::DiscoverLaw => {
                let already = self.is_busy(); // e.g. re-tracking: analysis follows by itself
                self.run_analysis();
                let started = already || self.is_busy();
                self.ai_reply_when(Waiting::Analysed, started, reply);
            }
            Action::SetWhatIf { gravity_scale, friction_scale } => {
                if let Some(g) = gravity_scale {
                    self.options.gravity_scale = g.clamp(0.0, 2.0);
                }
                if let Some(f) = friction_scale {
                    self.options.damping_scale = f.clamp(0.0, 2.0);
                }
                self.run_analysis();
                let started = self.is_busy();
                self.ai_reply_when(Waiting::Analysed, started, reply);
            }
            Action::SetTrainingWindow { seconds } => {
                self.options.train_until = seconds.max(1.0);
                self.run_analysis();
                let started = self.is_busy();
                self.ai_reply_when(Waiting::Analysed, started, reply);
            }
            Action::SetNeptune { mode, visible_fraction } => {
                let new_mode = match mode.as_str() {
                    "coupled_pair" => NeptuneMode::CoupledPair,
                    "hidden_hand" => NeptuneMode::HiddenHand,
                    _ => NeptuneMode::Off,
                };
                let new_keep = visible_fraction.map_or(self.tracker.keep_left, |v| v.clamp(0.3, 1.0));
                // Only a real change re-covers the hidden part; sending the same settings
                // again (the AI sometimes does) must not undo a reveal.
                if new_mode != self.options.neptune || (new_keep - self.tracker.keep_left).abs() > 1e-6 {
                    self.reveal = false;
                }
                self.options.neptune = new_mode;
                self.tracker.keep_left = new_keep;
                let _ = reply.send(Text(format!(
                    "Neptune Mode {mode}, camera sees the left {:.0}% of the frame. Now track, then discover_law.",
                    self.tracker.keep_left * 100.0
                )));
            }
            Action::Reveal { show } => {
                self.reveal = show;
                let _ = reply.send(Text(if show { "Revealed the hidden part." } else { "Hidden part covered again." }.into()));
            }
            Action::Play { playing } => {
                self.is_playing = playing && !self.frames.is_empty();
                self.last_frame_tick = Instant::now();
                let _ = reply.send(Text(if self.is_playing { "Playing." } else { "Paused." }.into()));
            }
            Action::GetResults => {
                let _ = reply.send(Text(self.results_summary()));
            }
        }
    }

    /// The AI found an object (box in frame pixels). Use it as the marker, the scale
    /// or the hidden card's pivot, exactly like the matching manual click tools.
    fn use_found_object(&mut self, bbox: [f64; 4], use_as: &str, width_mm: Option<f64>) -> String {
        let Some(frame) = self.frames.get(self.current_frame.min(self.frames.len().saturating_sub(1))).cloned() else {
            return "No video is open.".into();
        };
        let [x1, y1, x2, y2] = bbox;
        let (cx, cy) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0);
        match use_as {
            "scale" => {
                let mm = width_mm.unwrap_or(85.6);
                self.tracker.scale.p1 = Some([x1, cy]);
                self.tracker.scale.p2 = Some([x2, cy]);
                self.tracker.scale.real_distance_m = mm / 1000.0;
                self.scale_dist_mm = mm;
                format!("Scale set: {:.0} px = {mm} mm.", x2 - x1)
            }
            "hidden_pivot" => {
                self.hidden_pivot = Some([cx, cy]);
                format!("Hidden pivot set at ({cx:.0}, {cy:.0}).")
            }
            _ => {
                let colour = marker_colour(&frame, bbox);
                self.tracker.target_color = Some(colour);
                format!(
                    "Marker set to colour RGB({}, {}, {}) found at ({cx:.0}, {cy:.0}). Next: set the scale, then track.",
                    colour.r, colour.g, colour.b
                )
            }
        }
    }

    /// Everything the app knows right now, in plain words, for the AI to read.
    fn results_summary(&self) -> String {
        let r = &self.report;
        let mut s = String::new();
        if self.frames.is_empty() {
            return if self.rx_frames.is_some() { "A video is still loading.".into() } else { "No video is open.".into() };
        }
        let duration = self.frames.last().map_or(0.0, |f| f.time);
        let now = self.frames.get(self.current_frame).map_or(0.0, |f| f.time);
        s += &format!("Video: {} ({} frames, {duration:.1} s).\n", self.video_name, self.frames.len());
        s += &format!(
            "Playback: {} at frame {} ({now:.2} s).\n",
            if self.is_playing { "playing" } else { "paused" },
            self.current_frame
        );
        s += &format!("Marker colour: {}. ", if self.tracker.target_color.is_some() { "set" } else { "not set" });
        s += &match self.tracker.scale.pixel_distance() {
            Some(px) if self.tracker.scale.is_calibrated() => {
                format!("Scale: {px:.0} px = {:.1} mm.\n", self.tracker.scale.real_distance_m * 1000.0)
            }
            _ => "Scale: NOT set, so g is in pixel units (use find_object use_as='scale').\n".into(),
        };
        s += &format!(
            "Tracked frames: {}{}.\n",
            self.trajectory.t.len(),
            if self.is_busy() { " (a job is running right now)" } else { "" }
        );
        s += &format!(
            "Settings: training window {:.1} s, sparsity λ {:.2}, gravity x{:.2}, friction x{:.2}, speak replies {}.\n",
            self.options.train_until,
            self.options.threshold,
            self.options.gravity_scale,
            self.options.damping_scale,
            if self.assistant.speaks() { "on" } else { "off" }
        );
        s += &format!(
            "Neptune Mode: {:?}, visible part {:.0}%, hidden pivot {}, reveal {}.\n",
            self.options.neptune,
            self.tracker.keep_left * 100.0,
            if self.hidden_pivot.is_some() { "set" } else { "not set" },
            if self.reveal { "on" } else { "off" }
        );
        if r.law_typst.is_empty() {
            s += &format!("Law: {}\n", if r.law_plain.is_empty() { "not discovered yet" } else { &r.law_plain });
            return s;
        }
        s += &format!("Law: {}\n", r.law_plain);
        if let Some(c) = &r.conserved_typst {
            s += &format!("Conserved quantity: {}\n", equation::typst_to_unicode(c));
        }
        if let Some(l) = &r.leak_typst {
            s += &format!("Energy leak: {}\n", equation::typst_to_unicode(l));
        }
        s += &format!(
            "Training window: first {:.1} s. Ghost prediction from {:.1} s. What-if: gravity x{:.2}, friction x{:.2}.\n",
            self.options.train_until,
            r.ghost.t.first().copied().unwrap_or(0.0),
            self.options.gravity_scale,
            self.options.damping_scale
        );
        if self.options.neptune != NeptuneMode::Off {
            s += &format!(
                "Neptune Mode {:?}: hidden objects detected = {}, singular values {:.2?}. Hidden object rebuilt: {}.\n",
                self.options.neptune,
                r.hidden_count,
                &r.singular_values[..r.singular_values.len().min(4)],
                if r.hidden.is_some() { "yes" } else { "no" }
            );
        }
        s
    }

    /// Neptune Mode panel: badge, singular values, crop, hidden pivot, reveal.
    fn neptune_panel(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label(RichText::new("Neptune Mode").strong().size(14.0));
            ui.horizontal(|ui| {
                ComboBox::from_id_salt("neptune_combo")
                    .selected_text(match self.options.neptune {
                        NeptuneMode::Off => "Off",
                        NeptuneMode::CoupledPair => "Setup A: Coupled Pair",
                        NeptuneMode::HiddenHand => "Setup B: Hidden Hand",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.options.neptune, NeptuneMode::Off, "Off");
                        ui.selectable_value(&mut self.options.neptune, NeptuneMode::CoupledPair, "Setup A: Coupled Pair");
                        ui.selectable_value(&mut self.options.neptune, NeptuneMode::HiddenHand, "Setup B: Hidden Hand");
                    });
            });
            if self.options.neptune == NeptuneMode::Off {
                return;
            }

            // Crop: the camera only "sees" the left part. Moving it re-tracks.
            let mut keep = self.tracker.keep_left;
            ui.add(Slider::new(&mut keep, 0.3..=1.0).text("Visible part of frame"));
            if keep != self.tracker.keep_left && ui.input(|i| !i.pointer.any_down()) {
                self.tracker.keep_left = keep;
                self.reveal = false;
                self.run_tracking();
            } else if keep != self.tracker.keep_left {
                self.tracker.keep_left = keep; // still dragging: re-track on release
            }

            if self.options.neptune == NeptuneMode::CoupledPair {
                let picking = self.click_mode == ClickMode::HiddenPivot;
                if ui.selectable_label(picking, "📍 Click hidden card's pivot").clicked() {
                    self.click_mode = ClickMode::HiddenPivot;
                }
                if self.hidden_pivot.is_none() {
                    ui.label(RichText::new("Set the hidden pivot to draw the hidden card.").italics().color(Color32::GRAY));
                }
            }

            if !self.report.singular_values.is_empty() {
                let n = self.report.hidden_count;
                let colour = if n > 0 { Color32::from_rgb(255, 120, 200) } else { Color32::GRAY };
                ui.label(RichText::new(format!("Hidden objects detected: {n}")).size(18.0).strong().color(colour));
                plots::show_singular_values(ui, &self.report.singular_values);
            }
            let label = if self.reveal { "🙈 Hide again" } else { "👁 Reveal" };
            if ui.button(label).clicked() {
                self.reveal = !self.reveal;
            }
        });
    }
}

impl eframe::App for Video2LawApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.update_decoding();
        self.update_worker();
        for (action, reply) in self.assistant.poll() {
            self.perform_ai_action(action, reply);
        }
        if self.assistant.is_active() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
        if self.rx_frames.is_some() || self.is_busy() {
            ui.ctx().request_repaint(); // keep polling while loading or working
        }

        // A what-if slider moved: refresh the results (the engine is fast, Neptune ~1 s).
        if let Some(prev) = &self.analysed_with {
            if !same_options(prev, &self.options) && !self.is_busy() && ui.input(|i| !i.pointer.any_down()) {
                self.run_analysis();
            }
        }

        // Handle video playback timing
        if self.is_playing && !self.frames.is_empty() {
            let fps = self.meta.map(|m| m.fps).unwrap_or(30.0);
            let frame_dur = std::time::Duration::from_secs_f64(1.0 / fps);
            if self.last_frame_tick.elapsed() >= frame_dur {
                self.current_frame = (self.current_frame + 1) % self.frames.len();
                self.last_frame_tick = Instant::now();
            }
            ui.ctx().request_repaint();
        }

        // --- TOP TOOLBAR ---
        egui::Panel::top("top_toolbar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(RichText::new("Video2Law").color(Color32::from_rgb(0, 200, 255)).strong());
                ui.separator();

                if ui.button("📂 Open Video").clicked() {
                    if let Some(path) = FileDialog::new()
                        .add_filter("Videos", &["mp4", "mov", "avi", "mkv"])
                        .set_directory("clips")
                        .pick_file()
                    {
                        self.load_video_file(path);
                    }
                }

                if ui.add_enabled(!self.is_busy(), egui::Button::new("🎯 Track Motion")).clicked() {
                    self.run_tracking();
                }

                if ui.add_enabled(!self.is_busy(), egui::Button::new("⚡ Discover Law")).clicked() {
                    self.run_analysis();
                }
                if self.is_busy() {
                    ui.spinner();
                }

                ui.separator();

                // Click mode selector
                ui.label("Click tool:");
                if ui
                    .selectable_label(self.click_mode == ClickMode::PickMarker, "🎨 Pick Marker")
                    .clicked()
                {
                    self.click_mode = ClickMode::PickMarker;
                }
                if ui
                    .selectable_label(
                        self.click_mode == ClickMode::ScalePoint1 || self.click_mode == ClickMode::ScalePoint2,
                        "📏 Scale",
                    )
                    .clicked()
                {
                    self.click_mode = ClickMode::ScalePoint1;
                }
                ui.add(egui::DragValue::new(&mut self.scale_dist_mm).suffix(" mm").range(1.0..=5000.0));

                if let Some(color) = self.tracker.target_color {
                    ui.label("Marker:");
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), Sense::hover());
                    ui.painter().rect_filled(
                        rect,
                        3.0,
                        Color32::from_rgb(color.r, color.g, color.b),
                    );
                }
            });

            ui.add_space(2.0);
            ui.label(RichText::new(&self.status_message).size(12.0).color(Color32::LIGHT_GRAY));
            if !self.frames.is_empty() && !self.tracker.scale.is_calibrated() {
                ui.label(
                    RichText::new("⚠ Scale not set: click 📏 Scale, then two points a known distance apart. Until then g is meaningless.")
                        .size(12.0)
                        .color(Color32::from_rgb(255, 170, 60)),
                );
            }
        });

        // --- LEFT PANEL: AI ASSISTANT (chat + voice) ---
        egui::Panel::left("assistant_panel")
            .min_size(260.0)
            .default_size(300.0)
            .show(ui, |ui| self.assistant.ui(ui));

        // --- RIGHT PANEL: EQUATIONS, GRAPHS & KNOBS ---
        egui::Panel::right("right_panel")
            .min_size(420.0)
            .default_size(450.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(6.0);

                    // 1. Equation Display
                    equation::EquationView::show_law(ui, &self.report.law_typst, &self.report.law_plain);
                    ui.add_space(6.0);
                    equation::EquationView::show_conservation(
                        ui,
                        self.report.conserved_typst.as_deref(),
                        self.report.leak_typst.as_deref(),
                    );
                    ui.add_space(8.0);

                    // 2. Trajectory Plots
                    let ghost_ref = if !self.report.ghost.t.is_empty() {
                        Some(&self.report.ghost)
                    } else {
                        None
                    };
                    plots::show_x_t_plot(ui, &self.trajectory, ghost_ref, self.options.train_until);
                    ui.add_space(6.0);
                    plots::show_phase_portrait(ui, &self.trajectory, self.options.train_until);
                    ui.add_space(10.0);

                    // 3. Engine Physics Knobs & What-If
                    ui.group(|ui| {
                        ui.label(RichText::new("Physics Controls & What-If").strong().size(14.0));
                        ui.add_space(4.0);

                        let clip_end = self.trajectory.t.last().copied().unwrap_or(10.0).max(2.0);
                        ui.add(
                            Slider::new(&mut self.options.train_until, 1.0..=clip_end)
                                .text("Training window (s)"),
                        );
                        ui.add(Slider::new(&mut self.options.threshold, 0.01..=1.0).text("STLSQ Sparsity (λ)"));
                        ui.add(
                            Slider::new(&mut self.options.gravity_scale, 0.0..=2.0)
                                .text("Gravity Scale (1.0 Earth, 0.166 Moon)"),
                        );
                        ui.horizontal(|ui| {
                            if ui.button("🌍 Earth").clicked() {
                                self.options.gravity_scale = 1.0;
                            }
                            if ui.button("🌙 Moon").clicked() {
                                self.options.gravity_scale = 1.0 / 6.0;
                            }
                        });
                        ui.add(
                            Slider::new(&mut self.options.damping_scale, 0.0..=2.0)
                                .text("Friction Scale (0.0 no drag)"),
                        );
                    });
                    ui.add_space(8.0);

                    // 4. Neptune Mode
                    self.neptune_panel(ui);
                });
            });

        // --- CENTRAL PANEL: VIDEO PLAYER & OVERLAYS ---
        egui::CentralPanel::default().show(ui, |ui| {
            if self.frames.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("No video loaded.\nClick 'Open Video' above or place clips in the 'clips/' folder.");
                });
                return;
            }

            let frame_idx = self.current_frame.min(self.frames.len() - 1);
            let frame = Arc::clone(&self.frames[frame_idx]);

            // Update texture when current frame changes
            if self.video_texture_idx != Some(frame_idx) || self.video_texture.is_none() {
                let color_img = ColorImage::new(
                    [frame.width, frame.height],
                    frame
                        .rgb
                        .chunks_exact(3)
                        .map(|c| Color32::from_rgb(c[0], c[1], c[2]))
                        .collect(),
                );
                self.video_texture = Some(ui.ctx().load_texture(
                    "current_video_frame",
                    color_img,
                    TextureOptions::LINEAR,
                ));
                self.video_texture_idx = Some(frame_idx);
            }

            // Display video texture
            if let Some(texture) = &self.video_texture {
                let available_size = ui.available_size_before_wrap();
                let video_aspect = frame.width as f32 / frame.height as f32;
                let display_height = (available_size.x / video_aspect).min(available_size.y - 70.0).max(100.0);
                let display_width = display_height * video_aspect;

                let img_resp = ui.add(
                    egui::Image::new(texture)
                        .fit_to_exact_size(egui::vec2(display_width, display_height))
                        .sense(Sense::click()),
                );

                // Handle clicks on video for marker color, scale or hidden pivot
                if img_resp.clicked() {
                    if let Some(pos) = img_resp.interact_pointer_pos() {
                        let rel_x = (pos.x - img_resp.rect.min.x) / img_resp.rect.width();
                        let rel_y = (pos.y - img_resp.rect.min.y) / img_resp.rect.height();

                        let px = (rel_x * frame.width as f32).clamp(0.0, (frame.width - 1) as f32) as usize;
                        let py = (rel_y * frame.height as f32).clamp(0.0, (frame.height - 1) as f32) as usize;

                        match self.click_mode {
                            ClickMode::PickMarker => {
                                let offset = (py * frame.width + px) * 3;
                                let r = frame.rgb[offset];
                                let g = frame.rgb[offset + 1];
                                let b = frame.rgb[offset + 2];
                                self.tracker.target_color = Some(ColorRgb::new(r, g, b));
                                self.status_message = format!("Marker color set to RGB({}, {}, {}). Click 'Track Motion' to process.", r, g, b);
                            }
                            ClickMode::ScalePoint1 => {
                                self.tracker.scale.p1 = Some([px as f64, py as f64]);
                                self.tracker.scale.p2 = None;
                                self.click_mode = ClickMode::ScalePoint2;
                                self.status_message = "Click the other end of the known distance (Point 2).".to_string();
                            }
                            ClickMode::ScalePoint2 => {
                                self.tracker.scale.p2 = Some([px as f64, py as f64]);
                                self.tracker.scale.real_distance_m = self.scale_dist_mm / 1000.0;
                                self.click_mode = ClickMode::PickMarker;
                                self.status_message = format!(
                                    "Scale calibrated: {:.1} px = {:.1} mm. Track Motion again to use it.",
                                    self.tracker.scale.pixel_distance().unwrap_or(0.0),
                                    self.scale_dist_mm
                                );
                            }
                            ClickMode::HiddenPivot => {
                                self.hidden_pivot = Some([px as f64, py as f64]);
                                self.click_mode = ClickMode::PickMarker;
                                self.status_message = "Hidden pivot set.".to_string();
                            }
                        }
                    }
                }

                // Render Overlays on top of the image
                let overlay = overlay::FrameOverlay::new(
                    ui.painter(),
                    img_resp.rect,
                    frame.width,
                    frame.height,
                );
                let m_per_px = self.tracker.scale.meters_per_pixel();
                let origin = self.tracker.tracked_points.iter().find_map(|&p| p);

                // 0. Neptune crop: black out what the camera "can't see"
                if self.options.neptune != NeptuneMode::Off && !self.reveal {
                    overlay.draw_crop(self.tracker.keep_left);
                }

                // 1. Draw trajectory path trail
                overlay.draw_trajectory_trail(&self.tracker.tracked_points, frame_idx);

                // 2. Draw current tracked marker dot
                if let Some(Some([px, py])) = self.tracker.tracked_points.get(frame_idx) {
                    overlay.draw_current_marker(*px, *py);
                }

                // 3. Draw scale calibration line
                overlay.draw_scale_calibration(
                    self.tracker.scale.p1,
                    self.tracker.scale.p2,
                    self.tracker.scale.real_distance_m,
                );

                // 4. Draw predicted ghost if available
                if let Some(origin) = origin {
                    overlay.draw_ghost(&self.report.ghost, frame.time, origin, m_per_px);
                }

                // 5. Neptune: the rebuilt hidden object
                if let Some(hidden) = &self.report.hidden {
                    // Setup A gives the card relative to its own pivot; Setup B uses the
                    // same coordinates as the tracked object (relative to `origin`).
                    let anchor = match self.options.neptune {
                        NeptuneMode::CoupledPair => self.hidden_pivot,
                        _ => origin,
                    };
                    if let Some(anchor) = anchor {
                        let label = match self.options.neptune {
                            NeptuneMode::HiddenHand => "hidden hand",
                            _ => "hidden card",
                        };
                        overlay.draw_hidden(hidden, frame.time, anchor, m_per_px, label);
                    }
                }
            }

            // Playback Slider and Transport Controls
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let play_btn_text = if self.is_playing { "⏸ Pause" } else { "▶ Play" };
                if ui.button(play_btn_text).clicked() {
                    self.is_playing = !self.is_playing;
                    self.last_frame_tick = Instant::now();
                }

                let max_frames = self.frames.len().saturating_sub(1);
                ui.add(Slider::new(&mut self.current_frame, 0..=max_frames).text("Frame"));

                let cur_time = self.frames.get(self.current_frame).map(|f| f.time).unwrap_or(0.0);
                let total_time = self.frames.last().map(|f| f.time).unwrap_or(0.0);
                ui.label(format!("{:.2}s / {:.2}s ({} frames)", cur_time, total_time, self.frames.len()));
            });
        });
    }
}
