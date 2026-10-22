//! Conservation finder: discover energy without being told it exists (PLAN.md §3.3).
//!
//! We look for a mix of candidate terms  H = c₁·φ₁ + c₂·φ₂ + …  that stays constant
//! along the motion, i.e. dH/dt = Σ cₖ·dφₖ/dt = 0 at every frame.
//! Put dφₖ/dt for every frame in column k of a matrix D; then we need D·c ≈ 0.
//! The best c is the right singular vector of D with the smallest singular value.
//!
//! With friction nothing is exactly conserved, so we find H on a short early window
//! (about one swing), then fit how fast H drains over the whole clip: the "leak".

use crate::discover::stlsq;
use crate::library::{Style, Term, write_sum};
use crate::signal::PendulumSignal;
use faer::Mat;

/// Seconds at the start of the clip used to find H (about one swing of a lanyard).
const WINDOW_SECONDS: f64 = 1.5;

/// A candidate piece of the conserved quantity, and its rate of change.
struct Piece {
    term: Term,
    /// dφ/dt by the chain rule: ∂φ/∂θ·θ̇ + ∂φ/∂θ̇·θ̈. Arguments: (θ, θ̇, θ̈).
    rate: fn(f64, f64, f64) -> f64,
}

/// Candidate pieces. No constant term: "H = 1" is trivially conserved and useless.
fn pieces() -> Vec<Piece> {
    vec![
        Piece {
            term: Term { plain: "θ̇²", typst: "dot(theta)^2", eval: |_, v| v * v },
            rate: |_, v, a| 2.0 * v * a,
        },
        Piece {
            term: Term { plain: "cos θ", typst: "cos theta", eval: |q, _| q.cos() },
            rate: |q, v, _| -q.sin() * v,
        },
        Piece {
            term: Term { plain: "sin θ", typst: "sin theta", eval: |q, _| q.sin() },
            rate: |q, v, _| q.cos() * v,
        },
        Piece {
            term: Term { plain: "θ̇", typst: "dot(theta)", eval: |_, v| v },
            rate: |_, _, a| a,
        },
    ]
}

/// Candidate term for the leak dH/dt: friction drains energy at a rate b·θ̇².
/// (Air drag would be |θ̇|³, but on a 10-second clip it looks almost the same as θ̇²,
/// so offering both just splits the leak into a cancelling pair.)
fn leak_terms() -> Vec<Term> {
    vec![Term { plain: "θ̇²", typst: "dot(theta)^2", eval: |_, v| v * v }]
}

/// Leaks smaller than this are treated as "no leak". Lower than the law's λ because
/// gentle friction (b ≈ 0.1) must still show up.
const LEAK_THRESHOLD: f64 = 0.03;

/// The discovered conserved quantity H and its leak dH/dt = Σ leak coefficients × terms.
pub struct Conservation {
    terms: Vec<Term>,
    pub coefficients: Vec<f64>,
    leak_terms: Vec<Term>,
    pub leak: Vec<f64>,
}

/// Find the conserved quantity and its leak.
pub fn find_conserved(s: &PendulumSignal) -> Option<Conservation> {
    let pieces = pieces();
    let frame = |i: usize| (s.angle[i], s.velocity[i], s.acceleration[i]);

    // 1. D on the early window: D[i][k] = dφₖ/dt at frame i.
    let start = s.t.first()?;
    let rows = s.t.iter().take_while(|&&t| t - start <= WINDOW_SECONDS).count();
    if rows < pieces.len() * 3 {
        return None;
    }
    let d = Mat::from_fn(rows, pieces.len(), |i, k| {
        let (q, v, a) = frame(i);
        (pieces[k].rate)(q, v, a)
    });

    // Scale each column to unit size first, so a term isn't favoured just because
    // its numbers happen to be small.
    let scale: Vec<f64> = (0..pieces.len())
        .map(|k| (0..rows).map(|i| d[(i, k)].powi(2)).sum::<f64>().sqrt().max(1e-12))
        .collect();
    let d_scaled = Mat::from_fn(rows, pieces.len(), |i, k| d[(i, k)] / scale[k]);

    // 2. Like STLSQ, but with SVD: find the quietest direction, drop pieces that play
    //    almost no part in it, and repeat with the survivors.
    let mut active: Vec<usize> = (0..pieces.len()).collect();
    let mut direction = quietest_direction(&d_scaled, &active)?;
    loop {
        let survivors: Vec<usize> = active
            .iter()
            .zip(&direction)
            .filter(|&(&k, w)| k == 0 || w.abs() >= DROP_BELOW) // piece 0 (θ̇²) always stays
            .map(|(&k, _)| k)
            .collect();
        if survivors.len() == active.len() {
            break;
        }
        active = survivors;
        direction = quietest_direction(&d_scaled, &active)?;
    }

    // 3. Undo the column scaling and write it like a textbook energy: ½θ̇² + …
    let mut c = vec![0.0; pieces.len()];
    for (&k, w) in active.iter().zip(&direction) {
        c[k] = w / scale[k];
    }
    if c[0].abs() < 1e-9 {
        return None; // H doesn't involve motion at all, so it isn't an energy
    }
    let kinetic = c[0];
    for ck in c.iter_mut() {
        *ck *= 0.5 / kinetic;
    }

    // 4. Leak: dH/dt = Σ cₖ·dφₖ/dt over the whole clip, fitted against friction terms.
    let n = s.t.len();
    let dh_dt: Vec<f64> = (0..n)
        .map(|i| {
            let (q, v, a) = frame(i);
            pieces.iter().zip(&c).map(|(p, ck)| ck * (p.rate)(q, v, a)).sum()
        })
        .collect();
    let leak_terms = leak_terms();
    let theta = Mat::from_fn(n, leak_terms.len(), |i, k| (leak_terms[k].eval)(s.angle[i], s.velocity[i]));
    let leak = stlsq(&theta, &dh_dt, LEAK_THRESHOLD);

    Some(Conservation {
        terms: pieces.into_iter().map(|p| p.term).collect(),
        coefficients: c,
        leak_terms,
        leak,
    })
}

impl Conservation {
    /// The coefficient of the term printed as `plain` in H, or 0.0.
    pub fn coefficient(&self, plain: &str) -> f64 {
        find(&self.terms, &self.coefficients, plain)
    }

    /// The coefficient of the term printed as `plain` in the leak dH/dt, or 0.0.
    pub fn leak_coefficient(&self, plain: &str) -> f64 {
        find(&self.leak_terms, &self.leak, plain)
    }

    /// e.g. "E = ½θ̇² − 19.59·cos θ"
    pub fn conserved(&self, style: Style) -> String {
        write_sum("E =", &self.terms, &self.coefficients, style)
    }

    /// e.g. "dE/dt = −0.30·θ̇²"
    pub fn leak(&self, style: Style) -> String {
        let lhs = match style {
            Style::Plain => "dE/dt =",
            Style::Typst => "(dif E)/(dif t) =",
        };
        write_sum(lhs, &self.leak_terms, &self.leak, style)
    }
}

/// In the scaled problem, pieces whose weight is below this are dropped.
/// The weights form a unit vector, so 0.1 means "under 1 % of the total".
const DROP_BELOW: f64 = 0.1;

/// The right singular vector of the chosen columns with the smallest singular value:
/// the mix of those columns that is closest to zero along the whole window.
fn quietest_direction(d: &Mat<f64>, columns: &[usize]) -> Option<Vec<f64>> {
    let sub = Mat::from_fn(d.nrows(), columns.len(), |i, j| d[(i, columns[j])]);
    let svd = sub.thin_svd().ok()?;
    let last = columns.len() - 1; // faer sorts singular values largest first
    Some((0..columns.len()).map(|j| svd.V()[(j, last)]).collect())
}

fn find(terms: &[Term], coefficients: &[f64], plain: &str) -> f64 {
    terms.iter().zip(coefficients).find(|(t, _)| t.plain == plain).map_or(0.0, |(_, &c)| c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::pendulum_signal;
    use crate::testdata::{add_noise, pendulum};

    #[test]
    fn finds_energy_of_a_frictionless_pendulum() {
        // Energy per unit mass and length²: E = ½θ̇² − (g/L)·cos θ, with g/L = 9.81/0.5.
        let s = pendulum_signal(&pendulum(0.5, 0.8, 0.0, 10.0), f64::INFINITY).unwrap();
        let found = find_conserved(&s).unwrap();
        println!("{}   {}", found.conserved(Style::Plain), found.leak(Style::Plain));
        let g_over_l = -found.coefficient("cos θ");
        assert!((g_over_l - 19.62).abs() / 19.62 < 0.03, "{}", found.conserved(Style::Plain));
        assert_eq!(found.coefficient("sin θ"), 0.0);
        assert_eq!(found.leak_coefficient("θ̇²"), 0.0, "no friction, so no leak");
    }

    #[test]
    fn finds_energy_and_leak_of_a_damped_noisy_pendulum() {
        // With friction 0.3 the energy drains at dE/dt = −0.3·θ̇².
        let traj = add_noise(&pendulum(0.5, 0.8, 0.3, 10.0), 0.002);
        let s = pendulum_signal(&traj, f64::INFINITY).unwrap();
        let found = find_conserved(&s).unwrap();
        println!("{}   {}", found.conserved(Style::Plain), found.leak(Style::Plain));
        let g_over_l = -found.coefficient("cos θ");
        assert!((g_over_l - 19.62).abs() / 19.62 < 0.10, "{}", found.conserved(Style::Plain));
        let leak = found.leak_coefficient("θ̇²");
        assert!((leak + 0.3).abs() < 0.06, "{}", found.leak(Style::Plain));
    }
}
