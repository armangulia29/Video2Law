//! Engine demo in Simulation mode: Rapier scenes instead of video.
//! Run with: cargo run -p app --example demo
//!
//! Standalone on purpose, so it doesn't touch the real app files (app/src/).

use eframe::egui::{self, Color32};
use egui_plot::{Bar, BarChart, Legend, Line, Plot, VLine};
use engine::{NeptuneMode, Options, Report, Trajectory, analyse, hidden, signal, testdata};

#[derive(Clone, Copy, PartialEq)]
enum Scene {
    Swing,
    Damped,
    TwoCards,
    HiddenHand,
}

#[derive(Clone, PartialEq)]
struct Settings {
    scene: Scene,
    noise_mm: f64,
    train_until: f64,
    gravity_scale: f64,
    damping_scale: f64,
}

struct Demo {
    settings: Settings,
    last: Option<Settings>,
    seen: Trajectory,
    truth_hidden: Option<Trajectory>, // what the hidden object really did (for the reveal)
    report: Report,
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("Video2Law — engine demo").with_inner_size([1300.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Video2Law demo",
        options,
        Box::new(|_cc| {
            Ok(Box::new(Demo {
                settings: Settings { scene: Scene::Damped, noise_mm: 2.0, train_until: 5.0, gravity_scale: 1.0, damping_scale: 1.0 },
                last: None,
                seen: Trajectory::default(),
                truth_hidden: None,
                report: Report::default(),
            }))
        }),
    )
}

impl Demo {
    /// Build the chosen Rapier scene and run the whole engine on it.
    fn recompute(&mut self) {
        let s = &self.settings;
        let (clean, truth, neptune) = match s.scene {
            Scene::Swing => (testdata::pendulum(0.5, 0.8, 0.0, 10.0), None, NeptuneMode::Off),
            Scene::Damped => (testdata::pendulum(0.5, 0.8, 0.3, 10.0), None, NeptuneMode::Off),
            Scene::TwoCards => {
                let (left, right) = testdata::coupled_pendulums(0.5, 0.3, 3.0, 0.3, 40.0);
                let rel = Trajectory { t: right.t.clone(), x: right.x.iter().map(|x| x - 0.3).collect(), y: right.y.clone() };
                (left, Some(rel), NeptuneMode::CoupledPair)
            }
            Scene::HiddenHand => {
                let (keys, hand) = testdata::hidden_hand(0.5, hidden::STILL_SECONDS, 20.0);
                (keys, Some(hand), NeptuneMode::HiddenHand)
            }
        };
        self.seen = testdata::add_noise(&clean, s.noise_mm / 1000.0);
        self.truth_hidden = truth;
        let opts = Options {
            train_until: s.train_until,
            gravity_scale: s.gravity_scale,
            damping_scale: s.damping_scale,
            neptune,
            ..Default::default()
        };
        self.report = analyse(&self.seen, &opts);
        self.last = Some(self.settings.clone());
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.settings;
        ui.heading("Scene (Rapier)");
        ui.radio_value(&mut s.scene, Scene::Swing, "Single swing, no friction");
        ui.radio_value(&mut s.scene, Scene::Damped, "Damped swing");
        ui.radio_value(&mut s.scene, Scene::TwoCards, "Neptune A: two ID cards");
        ui.radio_value(&mut s.scene, Scene::HiddenHand, "Neptune B: hidden hand");
        ui.separator();
        ui.label("Tracking noise (mm)");
        ui.add(egui::Slider::new(&mut s.noise_mm, 0.0..=5.0));
        ui.label("Training window (s)");
        ui.add(egui::Slider::new(&mut s.train_until, 2.0..=9.0));
        ui.separator();
        ui.heading("What if…");
        ui.label("Gravity (1 = Earth, 0.17 = Moon)");
        ui.add(egui::Slider::new(&mut s.gravity_scale, 1.0 / 6.0..=1.0));
        ui.label("Friction (0 = none)");
        ui.add(egui::Slider::new(&mut s.damping_scale, 0.0..=1.0));
        if ui.button("Moon").clicked() {
            s.gravity_scale = 1.0 / 6.0;
        }
        if ui.button("Earth").clicked() {
            s.gravity_scale = 1.0;
        }
    }

    fn results(&self, ui: &mut egui::Ui) {
        let r = &self.report;
        ui.heading("Discovered");
        ui.monospace(format!("Law:        {}", r.law_plain));
        ui.monospace(format!("Conserved:  {}", r.conserved_typst.as_deref().unwrap_or("-")));
        ui.monospace(format!("Leak:       {}", r.leak_typst.as_deref().unwrap_or("-")));
        if !r.singular_values.is_empty() {
            ui.monospace(format!("Hidden objects detected: {}", r.hidden_count));
        }
        ui.separator();

        // Motion: what the camera saw, and the ghost predicted after the training window.
        ui.label("Sideways position x(t): seen (blue) vs ghost prediction (orange). Grey line = end of training.");
        let seen: Vec<[f64; 2]> = self.seen.t.iter().zip(&self.seen.x).map(|(t, x)| [*t, *x]).collect();
        let ghost: Vec<[f64; 2]> = r.ghost.t.iter().zip(&r.ghost.x).map(|(t, x)| [*t, *x]).collect();
        Plot::new("motion").height(220.0).legend(Legend::default()).show(ui, |p| {
            p.line(Line::new("seen", seen).color(Color32::from_rgb(70, 130, 220)));
            p.line(Line::new("ghost", ghost).color(Color32::from_rgb(240, 150, 40)).width(2.0));
            p.vline(VLine::new("training ends", self.settings.train_until).color(Color32::GRAY));
        });

        ui.columns(2, |cols| {
            // Phase portrait: angle against angular speed. Loop = energy kept; spiral = energy leaking.
            cols[0].label("Phase portrait θ vs θ̇ (loop = conserved, spiral = leaking)");
            if let Some(s) = signal::pendulum_signal(&self.seen, self.settings.train_until) {
                let pts: Vec<[f64; 2]> = s.angle.iter().zip(&s.velocity).map(|(a, v)| [*a, *v]).collect();
                Plot::new("phase").height(260.0).data_aspect(0.15).show(&mut cols[0], |p| {
                    p.line(Line::new("θ, θ̇", pts).color(Color32::from_rgb(120, 90, 200)));
                });
            }

            // Neptune: Hankel singular values, then the hidden object rebuilt vs the truth.
            if !r.singular_values.is_empty() {
                cols[1].label("Hankel singular values (2 per moving object)");
                let bars: Vec<Bar> = r.singular_values.iter().take(8).enumerate().map(|(i, v)| Bar::new(i as f64 + 1.0, *v)).collect();
                Plot::new("sv").height(110.0).show(&mut cols[1], |p| p.bar_chart(BarChart::new("σ", bars)));

                cols[1].label("Hidden object x(t): rebuilt (green) vs real (dashed grey)");
                Plot::new("hidden").height(130.0).legend(Legend::default()).show(&mut cols[1], |p| {
                    if let Some(truth) = &self.truth_hidden {
                        let pts: Vec<[f64; 2]> = truth.t.iter().zip(&truth.x).map(|(t, x)| [*t, *x]).collect();
                        p.line(Line::new("real", pts).color(Color32::GRAY).style(egui_plot::LineStyle::dashed_dense()));
                    }
                    if let Some(h) = &r.hidden {
                        let pts: Vec<[f64; 2]> = h.t.iter().zip(&h.x).map(|(t, x)| [*t, *x]).collect();
                        p.line(Line::new("rebuilt", pts).color(Color32::from_rgb(40, 170, 90)).width(2.0));
                    }
                });
            } else {
                cols[1].label("Pick a Neptune scene to see the hidden-object detector.");
            }
        });
    }
}

impl eframe::App for Demo {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.last.as_ref() != Some(&self.settings) {
            self.recompute();
        }
        egui::Panel::left("controls").show(ui, |ui| self.controls(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.results(ui));
        });
    }
}
