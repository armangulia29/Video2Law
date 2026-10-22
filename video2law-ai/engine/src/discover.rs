//! Law discovery with STLSQ, the solver behind SINDy (PLAN.md §3.2).
//!
//! We want the fewest terms ξ such that  q̈ ≈ Θ(q, q̇)·ξ.
//! 1. Least-squares fit with every term.
//! 2. Throw away terms whose coefficient is smaller than λ.
//! 3. Refit with the survivors. Repeat until nothing changes.

use crate::Trajectory;
use crate::library::{Style, Term, candidates, write_sum};
use crate::signal::pendulum_signal;
use faer::Mat;
use faer::prelude::*;

/// A discovered law: one coefficient per library term (0.0 = term not used).
pub struct Law {
    pub terms: Vec<Term>,
    pub coefficients: Vec<f64>,
}

/// Discover q̈ = f(q, q̇) from samples of q, q̇ and q̈ (all the same length).
pub fn discover(q: &[f64], v: &[f64], a: &[f64], lambda: f64) -> Law {
    let terms = candidates();
    // Θ: one row per frame, one column per candidate term.
    let theta = Mat::from_fn(q.len(), terms.len(), |i, k| (terms[k].eval)(q[i], v[i]));
    let coefficients = stlsq(&theta, a, lambda);
    Law { terms, coefficients }
}

/// Sequentially thresholded least squares. Returns one coefficient per column of `theta`.
pub fn stlsq(theta: &Mat<f64>, target: &[f64], lambda: f64) -> Vec<f64> {
    let mut active: Vec<usize> = (0..theta.ncols()).collect();
    let mut xi = vec![0.0; theta.ncols()];

    for _ in 0..10 {
        // Least squares using only the active terms.
        xi = vec![0.0; theta.ncols()];
        let fit = least_squares(theta, &active, target);
        for (&k, c) in active.iter().zip(fit) {
            xi[k] = c;
        }
        // Keep only terms whose coefficient is at least λ.
        let survivors: Vec<usize> = active.iter().copied().filter(|&k| xi[k].abs() >= lambda).collect();
        if survivors.len() == active.len() {
            break; // nothing removed, so we're done
        }
        active = survivors;
    }
    // Zero out anything that was dropped in the final round.
    for (k, c) in xi.iter_mut().enumerate() {
        if !active.contains(&k) {
            *c = 0.0;
        }
    }
    xi
}

/// Ordinary least squares on a subset of the columns of `theta`.
fn least_squares(theta: &Mat<f64>, columns: &[usize], target: &[f64]) -> Vec<f64> {
    if columns.is_empty() {
        return vec![];
    }
    let sub = Mat::from_fn(theta.nrows(), columns.len(), |i, j| theta[(i, columns[j])]);
    let rhs = Mat::from_fn(target.len(), 1, |i, _| target[i]);
    let sol = sub.qr().solve_lstsq(&rhs);
    (0..columns.len()).map(|j| sol[(j, 0)]).collect()
}

impl Law {
    /// The coefficient of the term printed as `plain`, or 0.0 if it isn't in the law.
    pub fn coefficient(&self, plain: &str) -> f64 {
        self.terms
            .iter()
            .zip(&self.coefficients)
            .find(|(t, _)| t.plain == plain)
            .map_or(0.0, |(_, &c)| c)
    }

    /// Gravity g in m/s², given the pendulum length L.
    ///
    /// For a pendulum θ̈ = −(g/L)·sin θ, so g = −L × (coefficient of sin θ).
    /// If the pivot we found is off by a small angle δ, the law becomes
    /// −(g/L)·sin(θ − δ) = −(g/L)·cos δ·sin θ + (g/L)·sin δ·cos θ:
    /// gravity is split between sin θ and cos θ, and the size of the pair is g/L.
    pub fn gravity(&self, length: f64) -> f64 {
        length * self.coefficient("sin θ").hypot(self.coefficient("cos θ"))
    }

    /// Unicode version, e.g. "θ̈ = −19.62·sin θ − 0.30·θ̇".
    pub fn plain(&self) -> String {
        write_sum("θ̈ =", &self.terms, &self.coefficients, Style::Plain)
    }

    /// Typst maths version, e.g. "dot.double(theta) = -19.62 sin theta - 0.30 dot(theta)".
    pub fn typst(&self) -> String {
        write_sum("dot.double(theta) =", &self.terms, &self.coefficients, Style::Typst)
    }
}

/// The full M2 pipeline for a swinging object, using only frames up to `train_until`:
/// positions → angle θ, θ̇, θ̈ (see `signal::pendulum_signal`) → law.
///
/// Returns the law and the pendulum length L (metres), or `None` if the clip is too short.
pub fn pendulum_law(traj: &Trajectory, train_until: f64, lambda: f64) -> Option<(Law, f64)> {
    let s = pendulum_signal(traj, train_until)?;
    let law = discover(&s.angle, &s.velocity, &s.acceleration, lambda);
    Some((law, s.length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::{add_noise, pendulum};

    fn discover_from_pendulum(traj: &Trajectory, lambda: f64) -> (Law, f64) {
        pendulum_law(traj, f64::INFINITY, lambda).expect("clip long enough")
    }

    #[test]
    fn recovers_g_from_clean_pendulum() {
        let traj = pendulum(0.5, 0.8, 0.0, 10.0);
        let (law, length) = discover_from_pendulum(&traj, 0.1);
        let g = law.gravity(length);
        println!("{}   g = {g:.3}", law.plain());
        assert!((g - 9.81).abs() < 0.2, "g = {g}");
    }

    #[test]
    fn recovers_g_and_friction_from_noisy_damped_pendulum() {
        // 2 mm of tracking noise on a 50 cm pendulum, with air friction.
        let traj = add_noise(&pendulum(0.5, 0.8, 0.3, 10.0), 0.002);
        let (law, length) = discover_from_pendulum(&traj, 0.1);
        let g = law.gravity(length);
        println!("{}   g = {g:.3}", law.plain());
        assert!((g - 9.81).abs() / 9.81 < 0.10, "g = {g}");
        assert!(law.coefficient("θ̇") < -0.1, "friction term missing: {}", law.plain());
    }
}
