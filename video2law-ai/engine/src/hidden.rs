//! Neptune Mode: detect and rebuild an object we can't see (PLAN.md §3.5–3.7).
//!
//! In 1846 Le Verrier saw Uranus wobble in a way its own physics couldn't explain,
//! and worked out where an unseen planet (Neptune) had to be. We do the same with one
//! tracked card: its motion carries the fingerprints of whatever it's tied to.

use crate::Trajectory;
use crate::signal::{HALF_WINDOW, PendulumSignal, SMOOTH_ORDER, pendulum_signal, savitzky_golay};
use faer::Mat;
use faer::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// A singular value counts as "real motion" if it's above this share of the largest.
/// Not 1 %: a big swing isn't a perfect sine wave, and its small overtone (3× the
/// frequency) shows up at 2–5 %. A real partner card shows up far above 10 %.
const SIGNIFICANT: f64 = 0.10;

// ── Detection: how many things are moving? (Hankel SVD) ──────────────────────────

/// Singular values of the Hankel matrix of `signal`, scaled so the largest is 1.
///
/// Row i of the Hankel matrix is the signal from frame i to frame i + delays − 1:
/// delayed copies of the same signal stacked up. Each oscillating part of the
/// motion (a sine wave) needs exactly 2 independent directions in this matrix, so
/// counting the big singular values counts the oscillators.
///
/// Each row spans a third of the clip (about 10 s). Two coupled cards swing at
/// almost the same rate, and only a long window can tell those two rhythms apart.
pub fn hankel_singular_values(signal: &[f64]) -> Vec<f64> {
    let delays = signal.len() / 3;
    if delays < 2 {
        return vec![];
    }
    // Remove the average first, otherwise the resting position counts as a "direction".
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let rows = signal.len() - delays + 1;
    let hankel = Mat::from_fn(rows, delays, |i, j| signal[i + j] - mean);
    let Ok(values) = hankel.singular_values() else { return vec![] };
    let largest = values[0].max(1e-12);
    values.iter().map(|v| v / largest).collect()
}

/// Hidden objects = (oscillators seen − 1), with 2 big singular values per oscillator.
pub fn hidden_count(singular_values: &[f64]) -> usize {
    let strong = singular_values.iter().filter(|&&v| v > SIGNIFICANT).count();
    strong.div_ceil(2).saturating_sub(1)
}

// ── Setup A: two ID cards on one cable (FFT normal modes) ────────────────────────

/// The two strongest rhythms in `signal` (sampled every `dt` s), as angular
/// frequencies ω in rad/s, slowest first. `None` if there aren't two clear peaks.
pub fn two_peaks(signal: &[f64], dt: f64) -> Option<(f64, f64)> {
    let n = signal.len();
    // Pad with zeros to 16× the length: same information, but a finer grid of
    // frequencies to read the peaks from.
    let size = (16 * n).next_power_of_two();
    let mean = signal.iter().sum::<f64>() / n as f64;
    let mut buffer: Vec<Complex<f64>> = (0..size)
        .map(|i| {
            if i >= n {
                return Complex::new(0.0, 0.0);
            }
            // Hann window: fade the ends to zero so the clip's cut-off doesn't smear the peaks.
            let fade = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
            Complex::new((signal[i] - mean) * fade, 0.0)
        })
        .collect();
    FftPlanner::new().plan_fft_forward(size).process(&mut buffer);
    let power: Vec<f64> = buffer[..size / 2].iter().map(|c| c.norm()).collect();

    // Local maxima, ignoring the very slow end (below 0.1 Hz).
    let bin_hz = 1.0 / (size as f64 * dt);
    let first = (0.1 / bin_hz).ceil() as usize;
    let mut peaks: Vec<(f64, f64)> = (first.max(1)..power.len() - 1)
        .filter(|&i| power[i] > power[i - 1] && power[i] >= power[i + 1])
        .map(|i| {
            // Fit a parabola through the peak and its neighbours to land between bins.
            let (a, b, c) = (power[i - 1], power[i], power[i + 1]);
            let shift = 0.5 * (a - c) / (a - 2.0 * b + c);
            let hz = (i as f64 + shift) * bin_hz;
            (2.0 * std::f64::consts::PI * hz, b)
        })
        .collect();
    peaks.sort_by(|p, q| q.1.total_cmp(&p.1)); // strongest first
    let (&(w1, p1), &(w2, p2)) = (peaks.first()?, peaks.get(1)?);
    if p2 < 0.05 * p1 {
        return None; // only one real rhythm: no partner to find
    }
    Some((w1.min(w2), w1.max(w2)))
}

/// Sharpen two rough frequencies (rad/s) by trying pairs close to them and keeping
/// the pair whose two sine waves best reproduce the signal (smallest leftover).
///
/// Why: a 30–40 s clip only holds 2–3 energy exchanges, so the two FFT peaks blur into
/// each other and their positions are off by a few %. That's too rough, because the
/// coupling k = ½(ω₂² − ω₁²) is a small difference of two big numbers.
pub fn sharpen_rhythms(signal: &[f64], t: &[f64], rough: (f64, f64)) -> (f64, f64) {
    let mut best = rough;
    // Coarse pass (±0.15 rad/s in 0.01 steps), then fine pass (±0.01 in 0.0005 steps).
    for (half_range, step) in [(0.15_f64, 0.01), (0.01, 0.0005)] {
        let centre = best;
        let mut best_leftover = f64::INFINITY;
        let n = (half_range / step).round() as i32;
        for i in -n..=n {
            for j in -n..=n {
                let w1 = centre.0 + i as f64 * step;
                let w2 = centre.1 + j as f64 * step;
                if w2 <= w1 + step {
                    continue;
                }
                let leftover = two_sine_leftover(signal, t, w1, w2);
                if leftover < best_leftover {
                    best_leftover = leftover;
                    best = (w1, w2);
                }
            }
        }
    }
    best
}

/// Fit signal ≈ a·cos ω₁t + b·sin ω₁t + c·cos ω₂t + d·sin ω₂t + e by least squares
/// and return the sum of squared leftovers.
fn two_sine_leftover(signal: &[f64], t: &[f64], w1: f64, w2: f64) -> f64 {
    let waves = |i: usize, k: usize| match k {
        0 => (w1 * t[i]).cos(),
        1 => (w1 * t[i]).sin(),
        2 => (w2 * t[i]).cos(),
        3 => (w2 * t[i]).sin(),
        _ => 1.0,
    };
    let lhs = Mat::from_fn(signal.len(), 5, waves);
    let rhs = Mat::from_fn(signal.len(), 1, |i, _| signal[i]);
    let c = lhs.qr().solve_lstsq(&rhs);
    (0..signal.len())
        .map(|i| {
            let fitted: f64 = (0..5).map(|k| c[(k, 0)] * waves(i, k)).sum();
            (signal[i] - fitted).powi(2)
        })
        .sum()
}

/// What Neptune Mode found about a coupled pair.
pub struct Partner {
    pub angle: Vec<f64>, // θ₂, the hidden card's swing angle, one per frame of the signal
    pub omega0: f64,     // ω₀, each card's own swing rate (rad/s)
    pub coupling: f64,   // k, how strongly the cable ties them together (1/s²)
}

/// Rebuild the hidden card's swing angle from the visible card's.
///
/// Two equal pendulums joined by a soft cable obey
///   θ̈₁ = −ω₀²·θ₁ − k·(θ₁ − θ₂)   ⇒   θ₂ = θ₁ + (θ̈₁ + ω₀²·θ₁) / k
/// The pair swings in two "normal modes": together (rate ω₁ = ω₀) and opposite
/// (rate ω₂, with ω₂² = ω₀² + 2k). The visible card is the sum of the two:
///   θ₁ = m₁(t) + m₂(t)
/// Put that into the formula above and it simplifies to
///   θ₂ = m₁(t) − m₂(t)
/// (the hidden card does the "together" swing too, but the "opposite" swing reversed).
/// So we split θ₁ into its two rhythms and flip one. Unlike the θ̈ formula, this
/// doesn't divide noisy accelerations by the small number k.
pub fn coupled_partner(s: &PendulumSignal) -> Option<Partner> {
    let dt = s.t[1] - s.t[0];
    let rough = two_peaks(&s.angle, dt)?;
    let (w1, w2) = sharpen_rhythms(&s.angle, &s.t, rough);

    // Fit θ₁ with both rhythms. Each gets cos and sin (any starting phase), and also
    // t·cos and t·sin, so its size can slowly shrink as friction takes energy away.
    let t0 = s.t[0];
    let column = |i: usize, k: usize| {
        let t = s.t[i] - t0;
        let w = if k < 4 { w1 } else { w2 };
        let ramp = if k % 4 < 2 { 1.0 } else { t };
        ramp * if k % 2 == 0 { (w * t).cos() } else { (w * t).sin() }
    };
    let n = s.angle.len();
    let lhs = Mat::from_fn(n, 8, column);
    let rhs = Mat::from_fn(n, 1, |i, _| s.angle[i]);
    let c = lhs.qr().solve_lstsq(&rhs);
    let mode = |i: usize, range: std::ops::Range<usize>| range.map(|k| c[(k, 0)] * column(i, k)).sum::<f64>();

    Some(Partner {
        angle: (0..n).map(|i| mode(i, 0..4) - mode(i, 4..8)).collect(),
        omega0: w1,
        coupling: 0.5 * (w2 * w2 - w1 * w1),
    })
}

// ── Setup B: keys on a hidden hand ───────────────────────────────────────────────

/// Seconds at the start of a Setup B clip when the hand must be held still.
pub const STILL_SECONDS: f64 = 3.0;

/// Rebuild the path of the hand holding the keys.
///
/// For small swings the keys are pulled towards the point under the hand, u(t):
///   ẍ = −ω₀²·(x − u)   ⇒   u = x + ẍ / ω₀²
/// No integration is needed, so the answer doesn't drift. ω₀ is measured during the
/// first `STILL_SECONDS`, when u is constant: there ẍ = −ω₀²·x + const, a straight-line fit.
pub fn hand_path(traj: &Trajectory) -> Option<Trajectory> {
    let still = pendulum_signal(traj, STILL_SECONDS)?; // the hand's height = pivot height
    let s = savitzky_golay(&traj.t, &traj.x, HALF_WINDOW, SMOOTH_ORDER);

    // Fit ẍ = a·x + b over the still window; then ω₀² = −a.
    let n_still = traj.t.iter().take_while(|&&t| t <= STILL_SECONDS).count();
    let rows = HALF_WINDOW..n_still.saturating_sub(HALF_WINDOW);
    if rows.len() < 10 {
        return None;
    }
    let lhs = Mat::from_fn(rows.len(), 2, |i, k| if k == 0 { s.value[rows.start + i] } else { 1.0 });
    let rhs = Mat::from_fn(rows.len(), 1, |i, _| s.acceleration[rows.start + i]);
    let fit = lhs.qr().solve_lstsq(&rhs);
    let w0_squared = -fit[(0, 0)];
    if w0_squared <= 0.0 {
        return None;
    }

    let keep = HALF_WINDOW..traj.t.len() - HALF_WINDOW;
    Some(Trajectory {
        t: traj.t[keep.clone()].to_vec(),
        x: keep.clone().map(|i| s.value[i] + s.acceleration[i] / w0_squared).collect(),
        y: keep.map(|_| still.pivot.1).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::{add_noise, coupled_pendulums, hidden_hand, pendulum};

    #[test]
    fn counts_hidden_objects() {
        let single = pendulum(0.5, 0.8, 0.0, 30.0);
        let damped = add_noise(&pendulum(0.5, 0.8, 0.3, 30.0), 0.002);
        let (pair, _) = coupled_pendulums(0.5, 0.3, 3.0, 0.3, 30.0);
        for (name, traj, expected) in [("single", &single, 0), ("damped", &damped, 0), ("coupled", &pair, 1)] {
            let sv = hankel_singular_values(&traj.x);
            println!("{name}: {:.4?}", &sv[..6]);
            assert_eq!(hidden_count(&sv), expected, "{name}");
        }
    }

    #[test]
    fn rebuilds_the_hidden_card() {
        // 40 s clip so the two FFT peaks are well separated.
        let (left, right) = coupled_pendulums(0.5, 0.3, 3.0, 0.3, 40.0);
        let seen = add_noise(&left, 0.002);
        let s = pendulum_signal(&seen, f64::INFINITY).unwrap();
        let rebuilt = coupled_partner(&s).expect("two peaks").angle;

        // Compare with the real right card's angle about its own pivot (0.3, 0).
        let mut worst: f64 = 0.0;
        let mut biggest: f64 = 0.0;
        for (i, &t) in s.t.iter().enumerate() {
            let j = (t * 30.0).round() as usize;
            let truth = (right.x[j] - 0.3).atan2(-right.y[j]);
            worst = worst.max((rebuilt[i] - truth).abs());
            biggest = biggest.max(truth.abs());
        }
        println!("hidden card: worst error {worst:.3} rad, biggest swing {biggest:.3} rad");
        assert!(worst < 0.15 * biggest, "worst error {worst:.3} rad");
    }

    #[test]
    fn rebuilds_the_hidden_hand() {
        let (keys, hand) = hidden_hand(0.5, STILL_SECONDS, 20.0);
        let seen = add_noise(&keys, 0.001);
        let rebuilt = hand_path(&seen).unwrap();
        let mut worst: f64 = 0.0;
        for (i, &t) in rebuilt.t.iter().enumerate() {
            let j = (t * 30.0).round() as usize;
            worst = worst.max((rebuilt.x[i] - hand.x[j]).abs());
        }
        println!("hidden hand: worst error {:.1} cm (hand sways ±7 cm)", worst * 100.0);
        assert!(worst < 0.02, "worst error {:.1} cm", worst * 100.0);
    }
}
