//! Signal preparation: turn the raw tracked positions into a clean angle θ(t)
//! with its velocity θ̇ and acceleration θ̈ (PLAN.md §3.1).

use crate::Trajectory;
use faer::Mat;
use faer::prelude::*;

/// A smoothed signal and its first two time derivatives, one value per frame.
pub struct Smoothed {
    pub value: Vec<f64>,        // q
    pub velocity: Vec<f64>,     // q̇
    pub acceleration: Vec<f64>, // q̈
}

/// Savitzky–Golay filter: smooth `y` and get its derivatives at the same time.
///
/// Around every sample we take a window of `2 * half_window + 1` neighbours and fit a
/// small polynomial (degree `order`) through them by least squares. The polynomial's
/// value, slope and curvature at the centre give q, q̇ and q̈. Noise averages out
/// because many points go into each fit.
pub fn savitzky_golay(t: &[f64], y: &[f64], half_window: usize, order: usize) -> Smoothed {
    let n = y.len();
    let width = (2 * half_window + 1).min(n);
    let mut out = Smoothed { value: vec![0.0; n], velocity: vec![0.0; n], acceleration: vec![0.0; n] };

    for i in 0..n {
        // Window centred on i, slid inwards near the ends of the clip.
        let start = i.saturating_sub(half_window).min(n - width);

        // Fit y ≈ c0 + c1·s + c2·s² + … where s = time measured from sample i.
        let powers = Mat::from_fn(width, order + 1, |r, k| (t[start + r] - t[i]).powi(k as i32));
        let values = Mat::from_fn(width, 1, |r, _| y[start + r]);
        let c = powers.qr().solve_lstsq(&values);

        // At s = 0: value = c0, slope = c1, curvature = 2·c2.
        out.value[i] = c[(0, 0)];
        out.velocity[i] = if order >= 1 { c[(1, 0)] } else { 0.0 };
        out.acceleration[i] = if order >= 2 { 2.0 * c[(2, 0)] } else { 0.0 };
    }
    out
}

/// Find the pivot of a swinging object by fitting a circle through its path.
/// Returns (pivot x, pivot y, radius), and the radius is the pendulum length L.
///
/// Step 1 (quick guess): every point on a circle obeys x² + y² = 2·a·x + 2·b·y + c,
/// which is *linear* in (a, b, c), so one least-squares solve gives a centre and radius.
/// On a short, noisy arc this guess comes out too small, so:
/// Step 2 (refine): nudge (a, b, r) to make every point's distance from the centre as
/// close to r as possible (Gauss–Newton: linearise, solve, repeat).
pub fn fit_circle(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len();
    let lhs = Mat::from_fn(n, 3, |i, k| match k {
        0 => 2.0 * x[i],
        1 => 2.0 * y[i],
        _ => 1.0,
    });
    let rhs = Mat::from_fn(n, 1, |i, _| x[i] * x[i] + y[i] * y[i]);
    let s = lhs.qr().solve_lstsq(&rhs);
    let (mut a, mut b) = (s[(0, 0)], s[(1, 0)]);
    let mut r = (s[(2, 0)] + a * a + b * b).sqrt();

    for _ in 0..20 {
        // Distance of each point from the current centre, and how it changes with a and b.
        let dist: Vec<f64> = (0..n).map(|i| (x[i] - a).hypot(y[i] - b).max(1e-12)).collect();
        let jacobian = Mat::from_fn(n, 3, |i, k| match k {
            0 => -(x[i] - a) / dist[i],
            1 => -(y[i] - b) / dist[i],
            _ => -1.0,
        });
        let error = Mat::from_fn(n, 1, |i, _| -(dist[i] - r));
        let step = jacobian.qr().solve_lstsq(&error);
        a += step[(0, 0)];
        b += step[(1, 0)];
        r += step[(2, 0)];
        if step[(0, 0)].abs() + step[(1, 0)].abs() + step[(2, 0)].abs() < 1e-10 {
            break;
        }
    }
    (a, b, r)
}

/// Swing angle θ of each point about the pivot (px, py), in radians; 0 = straight down.
///
/// Works whether y points up (metres, Rapier) or down (video pixels): we check which
/// side of the pivot the object hangs on and measure "down" towards that side.
pub fn swing_angle(x: &[f64], y: &[f64], px: f64, py: f64) -> Vec<f64> {
    let down = down_sign(y, py);
    x.iter().zip(y).map(|(x, y)| (x - px).atan2(down * (py - y))).collect()
}

/// +1 if the object hangs below the pivot with y pointing up, −1 if y points down.
fn down_sign(y: &[f64], py: f64) -> f64 {
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    if mean_y < py { 1.0 } else { -1.0 }
}

/// Smoothing window: 7 frames either side, fitted with a 4th-order polynomial.
/// Tuned on the Rapier pendulum at 30 fps (a cubic underestimates θ̈ by ~6 %).
pub(crate) const HALF_WINDOW: usize = 7;
pub(crate) const SMOOTH_ORDER: usize = 4;

/// A swinging object described by its angle instead of x and y.
pub struct PendulumSignal {
    pub t: Vec<f64>,            // seconds
    pub angle: Vec<f64>,        // θ, radians, 0 = straight down
    pub velocity: Vec<f64>,     // θ̇, rad/s
    pub acceleration: Vec<f64>, // θ̈, rad/s²
    pub length: f64,            // L, distance from pivot (same units as the trajectory)
    pub pivot: (f64, f64),      // where the lanyard hangs from
    pub down: f64,              // +1 if y points up (metres), −1 if y points down (pixels)
}

impl PendulumSignal {
    /// Turn an angle back into an (x, y) position, in the same units as the video.
    pub fn position(&self, angle: f64) -> (f64, f64) {
        let (px, py) = self.pivot;
        (px + self.length * angle.sin(), py - self.down * self.length * angle.cos())
    }
}

/// Positions → pivot and length → angle θ → smoothed θ, θ̇, θ̈.
///
/// Uses only frames up to `train_until`. Returns `None` if that's too short to smooth.
/// θ and θ̇ are also what the phase portrait plots.
pub fn pendulum_signal(traj: &Trajectory, train_until: f64) -> Option<PendulumSignal> {
    let n = traj.t.iter().take_while(|&&t| t <= train_until).count();
    if n < 4 * HALF_WINDOW {
        return None;
    }
    let (x, y) = (&traj.x[..n], &traj.y[..n]);
    let (px, py, length) = fit_circle(x, y);
    let angle = swing_angle(x, y, px, py);
    let s = savitzky_golay(&traj.t[..n], &angle, HALF_WINDOW, SMOOTH_ORDER);
    // Skip the ends, where the smoothing window is lopsided.
    let keep = HALF_WINDOW..n - HALF_WINDOW;
    Some(PendulumSignal {
        t: traj.t[keep.clone()].to_vec(),
        angle: s.value[keep.clone()].to_vec(),
        velocity: s.velocity[keep.clone()].to_vec(),
        acceleration: s.acceleration[keep].to_vec(),
        length,
        pivot: (px, py),
        down: down_sign(y, py),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_of_a_sine_wave() {
        // y = sin(2t) → y' = 2cos(2t), y'' = −4 sin(2t)
        let t: Vec<f64> = (0..300).map(|i| i as f64 / 30.0).collect();
        let y: Vec<f64> = t.iter().map(|t| (2.0 * t).sin()).collect();
        let s = savitzky_golay(&t, &y, 5, 3);
        for i in 10..290 {
            assert!((s.velocity[i] - 2.0 * (2.0 * t[i]).cos()).abs() < 0.01);
            assert!((s.acceleration[i] + 4.0 * (2.0 * t[i]).sin()).abs() < 0.05);
        }
    }

    #[test]
    fn circle_fit_finds_pivot_and_length() {
        let traj = crate::testdata::pendulum(0.5, 0.6, 0.0, 5.0);
        let (px, py, length) = fit_circle(&traj.x, &traj.y);
        assert!(px.abs() < 0.005 && py.abs() < 0.005, "pivot at ({px}, {py})");
        assert!((length - 0.5).abs() < 0.005, "length {length}");
    }
}
