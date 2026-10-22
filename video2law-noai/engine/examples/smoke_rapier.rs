use rapier2d::prelude::*;
use std::f32::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Smoke Test: rapier2d (Pendulum Period) ===");

    let mut world = PhysicsWorld::default();
    let g = 9.81;
    world.gravity = Vector::new(0.0, -g);

    let length = 1.0f32;
    let theta0 = 0.15f32; // ~8.6 degrees (small angle approximation valid)

    let bob_x = length * theta0.sin();
    let bob_y = -length * theta0.cos();

    let pivot = world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 0.0)));
    let bob = world.insert_body(RigidBodyBuilder::dynamic().translation(Vector::new(bob_x, bob_y)));
    world.insert_collider(ColliderBuilder::ball(0.05).density(1.0), Some(bob));

    let joint = RevoluteJointBuilder::new()
        .local_anchor1(Vector::new(0.0, 0.0))
        .local_anchor2(Vector::new(-bob_x, -bob_y));
    world.insert_impulse_joint(pivot, bob, joint);

    let dt = 1.0 / 120.0;
    world.integration_parameters.dt = dt;

    let mut times_crossing = Vec::new();
    let mut prev_x = bob_x;
    let total_steps = (6.0 / dt) as usize; // 6 seconds

    for step in 0..total_steps {
        world.step();
        let current_x = world.bodies[bob].translation().x;
        let t = step as f32 * dt;

        // Detect zero crossing from positive to negative
        if prev_x >= 0.0 && current_x < 0.0 {
            times_crossing.push(t);
        }
        prev_x = current_x;
    }

    println!("Zero crossing times: {:?}", times_crossing);
    assert!(times_crossing.len() >= 2, "Need at least 2 zero crossings to measure period");

    let measured_period = times_crossing[1] - times_crossing[0];
    let theoretical_period = 2.0 * PI * (length / g).sqrt();

    println!("Measured period: {:.3} s, Theoretical: {:.3} s", measured_period, theoretical_period);
    let diff = (measured_period - theoretical_period).abs();
    assert!(diff < 0.1, "Measured period deviates from theoretical: diff = {:.3}", diff);

    println!("PASS: rapier2d pendulum period");
    Ok(())
}
