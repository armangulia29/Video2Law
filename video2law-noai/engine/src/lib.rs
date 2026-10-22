//! Video2Law engine: pure maths, no GUI.
//!
//! This file is THE CONTRACT shared by both teammates (TECH_STACK.md §2.3).
//! The app only talks to the engine through `Trajectory`, `Options`, `Report` and `analyse`.
//! Don't change these types without telling your teammate first.

use serde::{Deserialize, Serialize};

pub mod conserve; // SVD conservation finder + leak
pub mod discover; // STLSQ law discovery
pub mod hidden; // Neptune Mode: Hankel SVD, FFT peaks, Setups A & B
pub mod library; // candidate terms (1, x, ẋ, sin x, …)
pub mod signal; // Savitzky–Golay smoothing + derivatives
pub mod simulate; // RK4 prediction + what-if
pub mod testdata; // Rapier-generated synthetic scenes

/// The tracked motion of one object, one entry per video frame.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Trajectory {
    pub t: Vec<f64>, // seconds
    pub x: Vec<f64>, // metres (after scale) — horizontal
    pub y: Vec<f64>, // metres — vertical
}

/// Which hidden-object reconstruction to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NeptuneMode {
    #[default]
    Off,
    CoupledPair, // Setup A: two ID cards, one cropped away
    HiddenHand,  // Setup B: keychain pulled by a hidden hand
}

/// Knobs the user controls from the UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Options {
    pub threshold: f64,       // STLSQ sparsity λ
    pub train_until: f64,     // seconds of data used for fitting; ghost predicts after this
    pub gravity_scale: f64,   // what-if: 1.0 Earth, 1/6 Moon
    pub damping_scale: f64,   // what-if: 0.0 = no friction
    pub neptune: NeptuneMode, // Off | CoupledPair | HiddenHand
}

impl Default for Options {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            train_until: 5.0,
            gravity_scale: 1.0,
            damping_scale: 1.0,
            neptune: NeptuneMode::Off,
        }
    }
}

/// Everything the engine found, ready for the UI to draw.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Report {
    pub law_typst: String, // e.g. "dot.double(theta) = -9.71 sin theta - 0.10 dot(theta)"
    pub law_plain: String, // Unicode fallback: "θ̈ = −9.71·sin θ − 0.10·θ̇"
    pub conserved_typst: Option<String>,
    pub leak_typst: Option<String>,
    pub ghost: Trajectory,          // RK4 prediction
    pub hidden: Option<Trajectory>, // Neptune Mode reconstruction
    pub singular_values: Vec<f64>,  // Hankel SVD, for the bar chart
    pub hidden_count: usize,        // "hidden objects: n"
}

/// Run the whole pipeline: smooth → STLSQ law → SVD energy → RK4 ghost → Neptune.
///
/// Law (M2), energy + leak (M3) and ghost (M4) are real; Neptune (M5) is still empty.
///
/// The ghost starts where the training window ends and runs to the end of the clip
/// plus 5 s, so the app can draw it next to the real card. Moving a what-if slider
/// only needs `analyse` again; it's fast.
pub fn analyse(traj: &Trajectory, opts: &Options) -> Report {
    let Some(signal) = signal::pendulum_signal(traj, opts.train_until) else {
        return Report {
            law_plain: "(clip too short to find a law)".to_string(),
            ..Default::default()
        };
    };
    let law = discover::discover(&signal.angle, &signal.velocity, &signal.acceleration, opts.threshold);
    let g = law.gravity(signal.length);
    let energy = conserve::find_conserved(&signal);
    let clip_end = traj.t.last().copied().unwrap_or(0.0);
    let ghost = simulate::ghost(&law, &signal, clip_end, opts.gravity_scale, opts.damping_scale);
    let mut report = Report {
        law_typst: law.typst(),
        law_plain: format!("{}    (g ≈ {g:.2} m/s², L = {:.2} m)", law.plain(), signal.length),
        conserved_typst: energy.as_ref().map(|e| e.conserved(library::Style::Typst)),
        leak_typst: energy.as_ref().map(|e| e.leak(library::Style::Typst)),
        ghost,
        ..Default::default()
    };
    if opts.neptune != NeptuneMode::Off {
        neptune(traj, opts.neptune, &mut report);
    }
    report
}

/// Neptune Mode (M5). Uses the whole clip, not just the training part: hidden rhythms
/// need as much video as possible (30 s or more for Setup A).
///
/// - `singular_values` / `hidden_count`: from the visible object's sideways motion x(t).
/// - Setup A (`CoupledPair`): `hidden` is the hidden card's position **relative to its
///   own pivot** (pivot at (0, 0), same units and y direction as the input). The app adds
///   the hidden card's pivot position (e.g. clicked by the user) to draw it.
/// - Setup B (`HiddenHand`): `hidden` is the hand's path in the same coordinates as the
///   input. The hand must be held still for the first `hidden::STILL_SECONDS` seconds.
fn neptune(traj: &Trajectory, mode: NeptuneMode, report: &mut Report) {
    report.singular_values = hidden::hankel_singular_values(&traj.x);
    report.hidden_count = hidden::hidden_count(&report.singular_values);
    report.hidden = match mode {
        NeptuneMode::CoupledPair => signal::pendulum_signal(traj, f64::INFINITY).and_then(|s| {
            let partner = hidden::coupled_partner(&s)?;
            let (length, down) = (s.length, s.down);
            Some(Trajectory {
                t: s.t.clone(),
                x: partner.angle.iter().map(|a| length * a.sin()).collect(),
                y: partner.angle.iter().map(|a| -down * length * a.cos()).collect(),
            })
        }),
        NeptuneMode::HiddenHand => hidden::hand_path(traj),
        NeptuneMode::Off => None,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyse_finds_the_pendulum_law() {
        let traj = testdata::pendulum(0.5, 0.8, 0.3, 10.0);
        let report = analyse(&traj, &Options::default());
        println!("{}\n{}\n{:?}\n{:?}", report.law_plain, report.law_typst, report.conserved_typst, report.leak_typst);
        assert!(report.law_typst.contains("sin theta"));
        assert!(report.conserved_typst.unwrap().contains("cos theta"));
        assert!(report.leak_typst.unwrap().contains("dot(theta)^2"));
        // Ghost: starts near the end of the 5 s training window, runs to 10 s + 5 s.
        assert!(report.ghost.t[0] > 4.0 && report.ghost.t[0] <= 5.0);
        assert!(*report.ghost.t.last().unwrap() > 14.9);
    }

    #[test]
    fn neptune_mode_through_analyse() {
        // Setup A: one hidden card, rebuilt relative to its own pivot.
        let (left, right) = testdata::coupled_pendulums(0.5, 0.3, 3.0, 0.3, 40.0);
        let opts = Options { neptune: NeptuneMode::CoupledPair, ..Default::default() };
        let report = analyse(&left, &opts);
        assert_eq!(report.hidden_count, 1);
        let hidden = report.hidden.unwrap();
        let i = 300; // compare one frame, 10 s or so in
        let j = (hidden.t[i] * 30.0).round() as usize;
        let miss = (hidden.x[i] - (right.x[j] - 0.3)).hypot(hidden.y[i] - right.y[j]);
        assert!(miss < 0.01, "hidden card {:.1} cm off", miss * 100.0);

        // Plain single pendulum: nothing hidden.
        let single = analyse(&testdata::pendulum(0.5, 0.8, 0.0, 30.0), &opts);
        assert_eq!(single.hidden_count, 0);

        // Setup B: the hand.
        let (keys, _) = testdata::hidden_hand(0.5, hidden::STILL_SECONDS, 20.0);
        let opts = Options { neptune: NeptuneMode::HiddenHand, ..Default::default() };
        assert!(analyse(&keys, &opts).hidden.is_some());
    }

    #[test]
    fn ghost_works_in_video_pixels_too() {
        // Same pendulum seen by a camera: 400 px per metre, y pointing down, pivot at (320, 50).
        let m = testdata::pendulum(0.5, 0.8, 0.0, 10.0);
        let px = Trajectory {
            t: m.t.clone(),
            x: m.x.iter().map(|x| 320.0 + 400.0 * x).collect(),
            y: m.y.iter().map(|y| 50.0 - 400.0 * y).collect(),
        };
        let report = analyse(&px, &Options::default());
        let i = px.t.iter().position(|&t| (t - report.ghost.t[30]).abs() < 1e-6).unwrap();
        let miss = (report.ghost.x[30] - px.x[i]).hypot(report.ghost.y[30] - px.y[i]);
        assert!(miss < 4.0, "ghost is {miss:.1} px off after 1 s");
    }
}
