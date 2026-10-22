use faer::{mat, Mat};
use faer::linalg::solvers::SolveLstsq;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Smoke Test: faer (Least Squares & SVD) ===");

    // 1. Least squares test:
    // A * x = b
    // [1, 1] [x0] = [6]
    // [1, 2] [x1] = [8]
    // [1, 3]        = [10]
    // Expected exact solution: x0 = 4, x1 = 2
    let a: Mat<f64> = mat![
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ];
    let b: Mat<f64> = mat![
        [6.0],
        [8.0],
        [10.0],
    ];

    let qr = a.col_piv_qr();
    let x = qr.solve_lstsq(&b);

    let x0 = x[(0, 0)];
    let x1 = x[(1, 0)];
    println!("Least squares solution: x0 = {:.4}, x1 = {:.4}", x0, x1);
    assert!((x0 - 4.0).abs() < 1e-6, "x0 should be 4.0, got {}", x0);
    assert!((x1 - 2.0).abs() < 1e-6, "x1 should be 2.0, got {}", x1);

    // 2. SVD test:
    let m: Mat<f64> = mat![
        [3.0, 0.0],
        [0.0, -5.0],
    ];
    let s = m.singular_values().map_err(|e| format!("{:?}", e))?;
    println!("Singular values: s0 = {:.4}, s1 = {:.4}", s[0], s[1]);
    assert!((s[0] - 5.0).abs() < 1e-6, "First singular value should be 5.0");
    assert!((s[1] - 3.0).abs() < 1e-6, "Second singular value should be 3.0");

    println!("PASS: faer least squares and SVD");
    Ok(())
}
