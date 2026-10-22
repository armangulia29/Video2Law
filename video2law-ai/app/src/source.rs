use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

/// Frames are shrunk to at most this many pixels (about 480×270).
/// All frames stay in memory: a 40 s clip at full 1080p would need ~7 GB,
/// at this size it needs ~0.5 GB. The sticker is still plenty of pixels to track.
const MAX_PIXELS: f64 = 130_000.0;

/// Metadata about a video stream.
#[derive(Debug, Clone, Copy)]
pub struct VideoMeta {
    pub width: usize,  // size of the decoded (shrunk, upright) frames
    pub height: usize,
    pub fps: f64,
}

/// A single decoded video frame in 24-bit RGB format.
pub struct VideoFrame {
    #[allow(dead_code)]
    pub index: usize,
    pub time: f64, // seconds from start
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Probes the video file using `ffprobe` to determine size and frame rate, and picks
/// the shrunk size we will decode at.
pub fn probe_video(path: &Path) -> Result<VideoMeta, String> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height,r_frame_rate:stream_side_data=rotation", "-of", "json"])
        .arg(path)
        .output()
        .map_err(|e| format!("Failed to run ffprobe: {}", e))?;
    if !output.status.success() {
        return Err(format!("ffprobe failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|e| format!("Unexpected ffprobe output: {e}"))?;
    let stream = &json["streams"][0];
    let mut width = stream["width"].as_u64().ok_or("No video width")? as f64;
    let mut height = stream["height"].as_u64().ok_or("No video height")? as f64;

    // Phone videos filmed upright are stored sideways with a "rotation" tag.
    // ffmpeg turns them upright when decoding, so swap width and height to match.
    let rotation = stream["side_data_list"][0]["rotation"].as_f64().unwrap_or(0.0);
    if (rotation.abs() as i64) % 180 == 90 {
        std::mem::swap(&mut width, &mut height);
    }

    // Frame rate: "30/1", "60000/1001" or "30".
    let fps_text = stream["r_frame_rate"].as_str().unwrap_or("30");
    let fps = match fps_text.split_once('/') {
        Some((num, den)) => num.parse::<f64>().unwrap_or(30.0) / den.parse::<f64>().unwrap_or(1.0).max(1.0),
        None => fps_text.parse().unwrap_or(30.0),
    };

    // Shrink to at most MAX_PIXELS, keeping the shape. Even sizes keep ffmpeg happy.
    let shrink = (MAX_PIXELS / (width * height)).sqrt().min(1.0);
    let even = |v: f64| ((v * shrink / 2.0).round() as usize * 2).max(2);
    Ok(VideoMeta { width: even(width), height: even(height), fps })
}

/// Spawns `ffmpeg` as a child process and decodes all frames on a background thread.
///
/// Sends each decoded frame through the returned `Receiver<VideoFrame>`.
/// An `Arc<AtomicBool>` cancellation token is returned to allow stopping the decoder thread early.
pub fn start_decoding(path: PathBuf, meta: VideoMeta) -> (Receiver<VideoFrame>, Arc<AtomicBool>) {
    let (tx, rx): (Sender<VideoFrame>, Receiver<VideoFrame>) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = Arc::clone(&cancel);

    thread::spawn(move || {
        let frame_size = meta.width * meta.height * 3;
        let mut child = match Command::new("ffmpeg")
            .arg("-i")
            .arg(&path)
            .args(["-vf", &format!("scale={}:{}", meta.width, meta.height)])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to spawn ffmpeg: {}", e);
                return;
            }
        };

        if let Some(mut stdout) = child.stdout.take() {
            let mut frame_idx = 0;
            loop {
                if cancel_clone.load(Ordering::Relaxed) {
                    let _ = child.kill();
                    break;
                }

                let mut buf = vec![0u8; frame_size];
                match stdout.read_exact(&mut buf) {
                    Ok(_) => {
                        let frame = VideoFrame {
                            index: frame_idx,
                            time: frame_idx as f64 / meta.fps,
                            width: meta.width,
                            height: meta.height,
                            rgb: buf,
                        };
                        frame_idx += 1;
                        if tx.send(frame).is_err() {
                            // Receiver dropped
                            let _ = child.kill();
                            break;
                        }
                    }
                    Err(_) => {
                        // EOF or error
                        break;
                    }
                }
            }
        }
        let _ = child.wait();
    });

    (rx, cancel)
}
