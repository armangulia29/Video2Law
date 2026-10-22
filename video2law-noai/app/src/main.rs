//! Video2Law desktop app: everything on screen. The maths lives in the `engine` crate.

mod equation;
mod overlay;
mod plots;
mod source;
mod tracker;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
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

            status_message: "Ready. Click 'Open Video' to load a swinging video clip.".to_string(),
        }
    }
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
            }
            Ok(WorkDone::Analysed(report)) => {
                self.report = report;
                self.worker = None;
                self.status_message = format!("Law discovered: {}", self.report.law_plain);
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                self.worker = None;
                self.status_message = "Background job failed.".to_string();
            }
        }
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
