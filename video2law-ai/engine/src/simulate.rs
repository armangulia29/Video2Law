//! Prediction: run the discovered law forward in time to draw the "ghost" (PLAN.md §3.4).
//!
//! The law tells us the acceleration θ̈ for any angle θ and speed θ̇. Knowing where the
//! card is and how fast it moves *now*, we can step forward a tiny time dt again and
//! again. RK4 (4th-order Runge–Kutta) checks the acceleration 4 times per step and
//! averages them, which keeps the ghost accurate over many swings.

use crate::Trajectory;
use crate::discover::Law;
use crate::signal::PendulumSignal;

/// RK4 steps per video frame. More steps = more accurate, and 4 is already plenty.
const STEPS_PER_FRAME: usize = 4;

/// Extra seconds the ghost keeps swinging after the end of the clip.
const EXTRA_SECONDS: f64 = 5.0;

/// The law's coefficients after the what-if sliders:
/// gravity terms (sin θ, cos θ) × `gravity_scale`, friction terms (anything with θ̇) × `damping_scale`.
pub fn what_if(law: &Law, gravity_scale: f64, damping_scale: f64) -> Vec<f64> {
    law.terms
        .iter()
        .zip(&law.coefficients)
        .map(|(term, &c)| match term.plain {
            "sin θ" | "cos θ" => c * gravity_scale,
            _ => c * damping_scale, // every other library term involves the speed θ̇
        })
        .collect()
}

/// θ̈ according to the law with the given coefficients.
fn acceleration(law: &Law, coefficients: &[f64], angle: f64, speed: f64) -> f64 {
    law.terms.iter().zip(coefficients).map(|(term, c)| c * (term.eval)(angle, speed)).sum()
}

/// One RK4 step of size dt from (θ, θ̇). Returns the new (θ, θ̇).
pub fn rk4_step(accel: impl Fn(f64, f64) -> f64, angle: f64, speed: f64, dt: f64) -> (f64, f64) {
    // Slopes at the start, twice in the middle, and at the end of the step.
    let (k1_angle, k1_speed) = (speed, accel(angle, speed));
    let (k2_angle, k2_speed) = {
        let (a, s) = (angle + 0.5 * dt * k1_angle, speed + 0.5 * dt * k1_speed);
        (s, accel(a, s))
    };
    let (k3_angle, k3_speed) = {
        let (a, s) = (angle + 0.5 * dt * k2_angle, speed + 0.5 * dt * k2_speed);
        (s, accel(a, s))
    };
    let (k4_angle, k4_speed) = {
        let (a, s) = (angle + dt * k3_angle, speed + dt * k3_speed);
        (s, accel(a, s))
    };
    // Weighted average of the four slopes (the middle ones count double).
    (
        angle + dt / 6.0 * (k1_angle + 2.0 * k2_angle + 2.0 * k3_angle + k4_angle),
        speed + dt / 6.0 * (k1_speed + 2.0 * k2_speed + 2.0 * k3_speed + k4_speed),
    )
}

/// The ghost: start from the last fitted frame and run the law until `until` seconds
/// (plus a few extra), one sample per video frame, as positions the app can draw.
pub fn ghost(law: &Law, s: &PendulumSignal, until: f64, gravity_scale: f64, damping_scale: f64) -> Trajectory {
    let coefficients = what_if(law, gravity_scale, damping_scale);
    let frame_dt = s.t[1] - s.t[0];
    let dt = frame_dt / STEPS_PER_FRAME as f64;

    let last = s.t.len() - 1;
    let (mut t, mut angle, mut speed) = (s.t[last], s.angle[last], s.velocity[last]);
    let mut out = Trajectory::default();
    while t <= until + EXTRA_SECONDS {
        let (x, y) = s.position(angle);
        out.t.push(t);
        out.x.push(x);
        out.y.push(y);
        for _ in 0..STEPS_PER_FRAME {
            (angle, speed) = rk4_step(|a, v| acceleration(law, &coefficients, a, v), angle, speed, dt);
        }
        t += frame_dt;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::discover;
    use crate::signal::pendulum_signal;
    use crate::testdata::{add_noise, pendulum};

    /// Train on the first 5 s, predict the next 5 s, and compare with what Rapier did.
    fn worst_ghost_error(damping: f64, noise: f64) -> f64 {
        let truth = pendulum(0.5, 0.8, damping, 10.0);
        let seen = add_noise(&truth, noise);
        let s = pendulum_signal(&seen, 5.0).unwrap();
        let law = discover(&s.angle, &s.velocity, &s.acceleration, 0.1);
        let g = ghost(&law, &s, 10.0, 1.0, 1.0);

        // Ghost frame i is at time g.t[i]; find the matching Rapier frame.
        let mut worst: f64 = 0.0;
        for i in 0..g.t.len() {
            let Some(j) = truth.t.iter().position(|&t| (t - g.t[i]).abs() < 1e-6) else { continue };
            worst = worst.max((g.x[i] - truth.x[j]).hypot(g.y[i] - truth.y[j]));
        }
        worst
    }

    #[test]
    fn ghost_stays_on_the_card() {
        // About 3.5 swings after training ends. A card is ~5 cm wide; stay within 3 cm.
        for (damping, noise) in [(0.0, 0.0), (0.3, 0.0), (0.3, 0.002)] {
            let err = worst_ghost_error(damping, noise);
            println!("damping {damping}, noise {noise}: worst ghost error {:.1} cm", err * 100.0);
            assert!(err < 0.03, "ghost drifted {:.1} cm", err * 100.0);
        }
    }

    #[test]
    fn moon_gravity_swings_slower() {
        // Small swings: period T = 2π√(L/g), so g ÷ 6 makes the period √6 ≈ 2.45× longer.
        let s = pendulum_signal(&pendulum(0.5, 0.2, 0.0, 6.0), f64::INFINITY).unwrap();
        let law = discover(&s.angle, &s.velocity, &s.acceleration, 0.1);
        let period = |gravity_scale: f64| {
            let g = ghost(&law, &s, 30.0, gravity_scale, 1.0);
            let ups: Vec<f64> = (1..g.x.len()).filter(|&i| g.x[i - 1] < 0.0 && g.x[i] >= 0.0).map(|i| g.t[i]).collect();
            (ups[ups.len() - 1] - ups[0]) / (ups.len() - 1) as f64
        };
        let ratio = period(1.0 / 6.0) / period(1.0);
        assert!((ratio - 6f64.sqrt()).abs() < 0.05, "period ratio {ratio}");
    }

    #[test]
    fn no_friction_keeps_swinging() {
        let s = pendulum_signal(&pendulum(0.5, 0.8, 0.3, 10.0), f64::INFINITY).unwrap();
        let law = discover(&s.angle, &s.velocity, &s.acceleration, 0.1);
        let amplitude = |damping_scale: f64| {
            let g = ghost(&law, &s, 30.0, 1.0, damping_scale);
            let tail = &g.x[g.x.len() - 60..]; // the last 2 seconds
            tail.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
        };
        // With friction the swing has nearly died by t = 35 s; without it, it hasn't.
        assert!(amplitude(0.0) > 5.0 * amplitude(1.0));
    }
}
