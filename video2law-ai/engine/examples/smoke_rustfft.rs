use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Smoke Test: rustfft (2 Hz Sine Peak) ===");

    let sample_rate = 30.0; // 30 Hz
    let n = 300; // 10 seconds of data -> 0.1 Hz frequency resolution
    let target_freq = 2.0; // 2 Hz

    let mut buffer: Vec<Complex<f64>> = (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let val = (2.0 * PI * target_freq * t).sin();
            Complex::new(val, 0.0)
        })
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    let mut max_mag = 0.0;
    let mut peak_bin = 0;
    for i in 1..(n / 2) {
        let mag = buffer[i].norm();
        if mag > max_mag {
            max_mag = mag;
            peak_bin = i;
        }
    }

    let peak_freq = peak_bin as f64 * sample_rate / n as f64;
    println!("Detected FFT peak frequency: {:.2} Hz (bin {}/{})", peak_freq, peak_bin, n);

    assert!(
        (peak_freq - target_freq).abs() < 0.15,
        "Expected peak at {:.2} Hz, got {:.2} Hz",
        target_freq,
        peak_freq
    );

    println!("PASS: rustfft 2 Hz peak");
    Ok(())
}
