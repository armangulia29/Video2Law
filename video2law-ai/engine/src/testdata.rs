//! Synthetic test scenes made with the Rapier physics engine.
//!
//! Rapier knows nothing about our maths, so if our engine rediscovers g from a Rapier
//! pendulum, we know the maths works before we ever try a real (noisy) video.

use crate::Trajectory;
use rapier2d::prelude::*;

/// Frames per second of the "camera" watching the scene (like a phone video).
pub const CAMERA_FPS: f64 = 30.0;
/// Physics steps per camera frame. Small physics steps keep the simulation accurate.
const STEPS_PER_FRAME: usize = 8;

/// A pendulum: a small ball on a rod of `length` metres, hanging from a pivot at (0, 0).
///
/// - `start_angle`: release angle in radians (0 = hanging straight down)
/// - `damping`: air friction; 0.0 = swings forever, ~0.3 = clearly dies down
/// - `seconds`: how long to record
///
/// Returns the ball's position once per camera frame, in metres, with y pointing up.
pub fn pendulum(length: f64, start_angle: f64, damping: f64, seconds: f64) -> Trajectory {
    let mut world = PhysicsWorld::new();
    world.gravity = Vec2::new(0.0, -9.81);
    world.integration_parameters.dt = (1.0 / (CAMERA_FPS * STEPS_PER_FRAME as f64)) as f32;

    // The pivot: a fixed body at the origin.
    let pivot = world.bodies.insert(RigidBodyBuilder::fixed());

    // The bob: a small dynamic ball, released at `start_angle`.
    let (s, c) = start_angle.sin_cos();
    let start = Vec2::new((length * s) as f32, (-length * c) as f32);
    let bob = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(start)
            .linear_damping(damping as f32)
            .can_sleep(false),
    );
    // Tiny radius so it behaves like a point mass (no extra spinning inertia).
    world
        .colliders
        .insert_with_parent(ColliderBuilder::ball(0.01), bob, &mut world.bodies);

    // The rod: a revolute (hinge) joint. Anchor 1 is the pivot itself;
    // anchor 2 is where the pivot sits as seen from the bob.
    let rod = RevoluteJointBuilder::new()
        .local_anchor1(Vec2::ZERO)
        .local_anchor2(-start);
    world.impulse_joints.insert(pivot, bob, rod, true);

    // Record one sample per camera frame.
    let frames = (seconds * CAMERA_FPS).round() as usize;
    let mut traj = Trajectory::default();
    for frame in 0..frames {
        let p = world.bodies[bob].translation();
        traj.t.push(frame as f64 / CAMERA_FPS);
        traj.x.push(p.x as f64);
        traj.y.push(p.y as f64);
        for _ in 0..STEPS_PER_FRAME {
            world.step();
        }
    }
    traj
}

/// Neptune Setup A: two equal pendulums whose bobs are joined by a soft spring
/// (the slack cable between the two lanyards). Only the left one is released, at
/// `start_angle`; energy then flows to the right one and back.
///
/// - `spacing`: distance between the two pivots, in metres
/// - `coupling`: spring stiffness (bigger = faster energy exchange)
///
/// Returns (visible left card, hidden right card).
pub fn coupled_pendulums(length: f64, spacing: f64, coupling: f64, start_angle: f64, seconds: f64) -> (Trajectory, Trajectory) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vec2::new(0.0, -9.81);
    world.integration_parameters.dt = (1.0 / (CAMERA_FPS * STEPS_PER_FRAME as f64)) as f32;

    let (l, d) = (length as f32, spacing as f32);
    let (s, c) = (start_angle.sin() as f32, start_angle.cos() as f32);
    let starts = [Vec2::new(l * s, -l * c), Vec2::new(d, -l)];
    let pivots = [Vec2::ZERO, Vec2::new(d, 0.0)];

    let mut bobs = Vec::new();
    for (start, pivot_at) in starts.iter().zip(&pivots) {
        let pivot = world.bodies.insert(RigidBodyBuilder::fixed().translation(*pivot_at));
        let bob = world.bodies.insert(RigidBodyBuilder::dynamic().translation(*start).can_sleep(false));
        world.colliders.insert_with_parent(ColliderBuilder::ball(0.01), bob, &mut world.bodies);
        let rod = RevoluteJointBuilder::new().local_anchor1(Vec2::ZERO).local_anchor2(*pivot_at - *start);
        world.impulse_joints.insert(pivot, bob, rod, true);
        bobs.push(bob);
    }
    // The cable: a spring between the two bobs, relaxed when both hang straight down.
    // Acceleration-based, so `coupling` doesn't depend on how heavy the bobs are.
    let cable = SpringJointBuilder::new(d, coupling as f32, 0.0).spring_model(MotorModel::AccelerationBased);
    world.impulse_joints.insert(bobs[0], bobs[1], cable, true);

    let frames = (seconds * CAMERA_FPS).round() as usize;
    let (mut left, mut right) = (Trajectory::default(), Trajectory::default());
    for frame in 0..frames {
        let t = frame as f64 / CAMERA_FPS;
        for (traj, &bob) in [&mut left, &mut right].into_iter().zip(&bobs) {
            let p = world.bodies[bob].translation();
            traj.t.push(t);
            traj.x.push(p.x as f64);
            traj.y.push(p.y as f64);
        }
        for _ in 0..STEPS_PER_FRAME {
            world.step();
        }
    }
    (left, right)
}

/// Neptune Setup B: keys hanging from a hand that we can't see.
/// The hand holds still for the first `still_seconds` (so the keys' own swing rate
/// can be measured), then sways gently from side to side.
///
/// Returns (visible keys, hidden hand). The hand's y is always 0.
pub fn hidden_hand(length: f64, still_seconds: f64, seconds: f64) -> (Trajectory, Trajectory) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vec2::new(0.0, -9.81);
    let step_dt = 1.0 / (CAMERA_FPS * STEPS_PER_FRAME as f64);
    world.integration_parameters.dt = step_dt as f32;

    // Hand path: still, then a slow sway made of two gentle waves (5 cm and 2 cm).
    let hand_x = |t: f64| {
        let s = (t - still_seconds).max(0.0);
        let ramp = (s / 1.0).min(1.0); // ease in over 1 s so the start isn't a jerk
        ramp * (0.05 * (2.0 * std::f64::consts::PI * 0.25 * s).sin() + 0.02 * (2.0 * std::f64::consts::PI * 0.4 * s).sin())
    };

    let hand = world.bodies.insert(RigidBodyBuilder::kinematic_position_based());
    let start_angle: f64 = 0.15; // a small swing, so there's something to measure while the hand is still
    let start = Vec2::new((length * start_angle.sin()) as f32, (-length * start_angle.cos()) as f32);
    let keys = world.bodies.insert(RigidBodyBuilder::dynamic().translation(start).can_sleep(false));
    world.colliders.insert_with_parent(ColliderBuilder::ball(0.01), keys, &mut world.bodies);
    let rod = RevoluteJointBuilder::new().local_anchor1(Vec2::ZERO).local_anchor2(-start);
    world.impulse_joints.insert(hand, keys, rod, true);

    let frames = (seconds * CAMERA_FPS).round() as usize;
    let (mut seen, mut hidden) = (Trajectory::default(), Trajectory::default());
    let mut t = 0.0;
    for _ in 0..frames {
        let p = world.bodies[keys].translation();
        seen.t.push(t);
        seen.x.push(p.x as f64);
        seen.y.push(p.y as f64);
        hidden.t.push(t);
        hidden.x.push(hand_x(t));
        hidden.y.push(0.0);
        for _ in 0..STEPS_PER_FRAME {
            t += step_dt;
            world.bodies[hand].set_next_kinematic_translation(Vec2::new(hand_x(t) as f32, 0.0));
            world.step();
        }
    }
    (seen, hidden)
}

/// Add tracking noise (like a wobbly colour tracker) of up to ±`amount` metres.
///
/// Uses a tiny hand-written random generator so every test run gets the same noise.
pub fn add_noise(traj: &Trajectory, amount: f64) -> Trajectory {
    let mut seed: u64 = 12345;
    let mut random = move || {
        // Linear congruential generator → a number in [-1, 1)
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    Trajectory {
        t: traj.t.clone(),
        x: traj.x.iter().map(|x| x + amount * random()).collect(),
        y: traj.y.iter().map(|y| y + amount * random()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pendulum_period_matches_textbook() {
        // Small swing, so the textbook period T = 2π√(L/g) holds.
        let length = 0.5;
        let traj = pendulum(length, 0.1, 0.0, 10.0);

        // Count upward zero crossings of x to measure the period.
        let crossings: Vec<f64> = (1..traj.x.len())
            .filter(|&i| traj.x[i - 1] < 0.0 && traj.x[i] >= 0.0)
            .map(|i| traj.t[i])
            .collect();
        let measured = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
        let expected = 2.0 * std::f64::consts::PI * (length / 9.81).sqrt();
        assert!(
            (measured - expected).abs() / expected < 0.03,
            "period {measured:.3} s, expected {expected:.3} s"
        );
    }

    #[test]
    fn rod_length_stays_constant() {
        let traj = pendulum(0.5, 1.0, 0.0, 5.0);
        for (x, y) in traj.x.iter().zip(&traj.y) {
            let r = (x * x + y * y).sqrt();
            assert!((r - 0.5).abs() < 0.005, "rod length drifted to {r}");
        }
    }
}
