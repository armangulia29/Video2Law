use std::io::Read;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Smoke Test: FFmpeg Pipe ===");
    let clip_path = "clips/test.mp4";
    if !std::path::Path::new(clip_path).exists() {
        return Err(format!("Clip not found at {}", clip_path).into());
    }

    let width = 320;
    let height = 240;
    let frame_bytes = width * height * 3;

    let mut child = Command::new("ffmpeg")
        .args([
            "-i",
            clip_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    let mut stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let mut buffer = vec![0u8; frame_bytes];
    stdout.read_exact(&mut buffer)?;

    let _ = child.kill();
    let _ = child.wait();

    println!("Successfully read {} bytes ({}x{} RGB) from FFmpeg pipe!", buffer.len(), width, height);
    println!("PASS: ffmpeg pipe");
    Ok(())
}
