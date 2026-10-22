//! Speech-to-text check: Windows speaks a sentence into a WAV, Whisper writes it back.
//! Run with: cargo run -p ai --example listen_check

use std::path::Path;
use std::process::Command;

const WAV: &str = "listen_check.wav"; // written to the current folder, deleted after
const SENTENCE: &str = "Open the swing video and find the value of g.";

fn main() {
    // 1. Make the test recording with Windows' built-in voice.
    let script = format!(
        "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
         $s.SetOutputToWaveFile((Join-Path (Get-Location) '{WAV}')); $s.Speak('{SENTENCE}'); $s.Dispose()"
    );
    let ok = Command::new("powershell").args(["-NoProfile", "-Command", &script]).status().map(|s| s.success());
    assert!(matches!(ok, Ok(true)), "couldn't make the test WAV");

    // 2. Read it (16-bit PCM) and bring it to 16 kHz mono.
    let samples = read_wav_16k(Path::new(WAV));
    let _ = std::fs::remove_file(WAV);
    println!("audio: {:.1} s", samples.len() as f32 / 16_000.0);

    // 3. Transcribe.
    let started = std::time::Instant::now();
    let mut ears = ai::listen::Ears::load(&ai::models_dir().join("whisper-base-en")).expect("load Whisper");
    println!("Whisper loaded in {:.1} s", started.elapsed().as_secs_f32());
    let started = std::time::Instant::now();
    let text = ears.transcribe(&samples).expect("transcribe");
    println!("said:  {SENTENCE}\nheard: {text}\n({:.1} s)", started.elapsed().as_secs_f32());
}

/// Minimal WAV reader for 16-bit PCM: find the "fmt " and "data" chunks.
fn read_wav_16k(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav");
    let (mut rate, mut channels, mut pos) = (16_000u32, 1u16, 12usize);
    let mut samples = Vec::new();
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        if id == b"fmt " {
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
        } else if id == b"data" {
            samples = body.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();
        }
        pos += 8 + size + (size & 1);
    }
    // Mono, then straight-line resampling to 16 kHz.
    let mono: Vec<f32> = samples.chunks(channels as usize).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect();
    let n = (mono.len() as f64 * 16_000.0 / rate as f64) as usize;
    (0..n)
        .map(|i| {
            let p = i as f64 * rate as f64 / 16_000.0;
            let j = (p as usize).min(mono.len() - 1);
            let k = (j + 1).min(mono.len() - 1);
            mono[j] + (mono[k] - mono[j]) * (p - j as f64) as f32
        })
        .collect()
}
