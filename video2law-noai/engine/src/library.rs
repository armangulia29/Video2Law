//! The library of candidate terms Θ(q, q̇) for law discovery (PLAN.md §3.2).
//!
//! The law is assumed to be  q̈ = ξ₁·term₁ + ξ₂·term₂ + …  and STLSQ picks the few
//! terms that matter. Here q is the swing angle θ and q̇ its angular velocity.

/// One candidate term, with how to compute it and how to print it.
pub struct Term {
    pub plain: &'static str, // Unicode, e.g. "sin θ"
    pub typst: &'static str, // Typst maths, e.g. "sin theta"
    pub eval: fn(f64, f64) -> f64,
}

/// Unicode for on-screen fallback text, or Typst maths for the rendered equation.
#[derive(Clone, Copy)]
pub enum Style {
    Plain,
    Typst,
}

/// Write `lhs c₁·term₁ + c₂·term₂ …`, skipping terms whose coefficient is 0.0.
/// A coefficient of exactly ½ is written as a fraction, as in kinetic energy ½θ̇².
pub fn write_sum(lhs: &str, terms: &[Term], coefficients: &[f64], style: Style) -> String {
    let (minus, times, half) = match style {
        Style::Plain => ("−", "·", "½"),
        Style::Typst => ("-", " ", "1/2 "),
    };
    let mut out = lhs.to_string();
    let mut first = true;
    for (term, &c) in terms.iter().zip(coefficients) {
        if c == 0.0 {
            continue;
        }
        let sign = match (c < 0.0, first) {
            (true, true) => format!(" {minus}"),
            (true, false) => format!(" {minus} "),
            (false, true) => " ".to_string(),
            (false, false) => " + ".to_string(),
        };
        let name = match style {
            Style::Plain => term.plain,
            Style::Typst => term.typst,
        };
        let body = if term.plain == "1" {
            format!("{:.2}", c.abs())
        } else if c.abs() == 0.5 {
            format!("{half}{name}")
        } else {
            format!("{:.2}{times}{name}", c.abs())
        };
        out += &(sign + &body);
        first = false;
    }
    if first {
        out += " 0";
    }
    out
}

/// Every candidate term we try.
///
/// Angles enter physics through sin and cos, so we offer trig functions of θ, not
/// powers θ, θ², θ³. Those powers are almost the same curves as sin θ for a normal
/// swing (sin θ ≈ θ − θ³/6), and the regression can't tell them apart: it then splits
/// the answer across look-alike terms and the law comes out messy.
/// For speed we offer θ̇ (smooth friction) and θ̇·|θ̇| (air drag).
///
/// No constant term: for small swings cos θ ≈ 1, so "1" and "cos θ" are look-alikes
/// and would appear as a cancelling pair. cos θ alone is kept because it has a real
/// meaning: it shows up when the pivot we found is a little off to one side.
pub fn candidates() -> Vec<Term> {
    vec![
        Term { plain: "sin θ", typst: "sin theta", eval: |q, _| q.sin() },
        Term { plain: "cos θ", typst: "cos theta", eval: |q, _| q.cos() },
        Term { plain: "θ̇", typst: "dot(theta)", eval: |_, v| v },
        Term { plain: "θ̇·|θ̇|", typst: "dot(theta) abs(dot(theta))", eval: |_, v| v * v.abs() },
        Term { plain: "θ̇²", typst: "dot(theta)^2", eval: |_, v| v * v },
    ]
}
