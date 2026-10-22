//! Robustness sweep: run the law + energy finders on many Rapier pendulums and grade them.
//! Run with: cargo test -p engine --test sweep -- --nocapture

use engine::conserve::find_conserved;
use engine::discover::discover;
use engine::library::Style;
use engine::signal::pendulum_signal;
use engine::testdata;

#[test]
fn sweep() {
    let (mut total, mut g_ok, mut e_ok, mut leak_ok) = (0, 0, 0, 0);
    for length in [0.2, 0.3, 0.5, 0.8] {
        for angle in [0.3, 0.6, 1.0] {
            for damping in [0.0, 0.1, 0.3] {
                for noise in [0.0, 0.002, 0.005] {
                    total += 1;
                    let traj = testdata::add_noise(&testdata::pendulum(length, angle, damping, 10.0), noise);
                    let s = pendulum_signal(&traj, f64::INFINITY).unwrap();
                    let law = discover(&s.angle, &s.velocity, &s.acceleration, 0.1);
                    let g = law.gravity(s.length);
                    let true_g_over_l = 9.81 / length;

                    // Law: g within 10 %.
                    let pass_g = (g - 9.81).abs() / 9.81 < 0.10;

                    // Energy: ½θ̇² − (g/L)·cos θ within 10 %, nothing else.
                    let found = find_conserved(&s);
                    let pass_e = found.as_ref().is_some_and(|f| {
                        (-f.coefficient("cos θ") - true_g_over_l).abs() / true_g_over_l < 0.10
                            && f.coefficient("sin θ") == 0.0
                            && f.coefficient("θ̇") == 0.0
                    });

                    // Leak: −damping·θ̇² (within 30 %, or ±0.03 when there is no friction).
                    let pass_leak = found.as_ref().is_some_and(|f| {
                        (f.leak_coefficient("θ̇²") + damping).abs() <= 0.3 * damping + 0.03
                            && f.leak_coefficient("|θ̇|³") == 0.0
                    });

                    g_ok += pass_g as usize;
                    e_ok += pass_e as usize;
                    leak_ok += pass_leak as usize;
                    if !(pass_g && pass_e && pass_leak) {
                        let (e, leak) = found.map_or(("-".into(), "-".into()), |f| {
                            (f.conserved(Style::Plain), f.leak(Style::Plain))
                        });
                        println!(
                            "FAIL L={length} θ0={angle} d={damping} noise={noise}: [g {}{}] {}  g={g:.2} L={:.3} | [E {}] {e} | [leak {}] {leak}",
                            if pass_g { "ok" } else { "BAD" }, "", law.plain(), s.length,
                            if pass_e { "ok" } else { "BAD" },
                            if pass_leak { "ok" } else { "BAD" },
                        );
                    }
                }
            }
        }
    }
    println!("\n{total} scenes: g ok {g_ok}, energy ok {e_ok}, leak ok {leak_ok}");
    // Lock in today's quality (106 / 108 / 106) so a later change can't quietly break it.
    // The few misses are a 20 cm lanyard, tiny swing and 5 mm tracking noise.
    assert!(g_ok >= 104 && e_ok >= 104 && leak_ok >= 104, "the sweep got worse");
}
