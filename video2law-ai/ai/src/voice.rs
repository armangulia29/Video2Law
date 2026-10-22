//! Microphone in (`cpal`), Windows voice out (`tts`).

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

/// Whisper listens at 16 000 samples per second.
const WHISPER_RATE: f64 = 16_000.0;

/// Push-to-talk recording. Created on the UI thread when the talk button goes down,
/// stopped when it comes up. (`cpal` streams must stay on the thread that made them.)
pub struct Mic {
    stream: cpal::Stream,
    samples: Arc<Mutex<Vec<f32>>>, // interleaved, as the microphone delivers them
    rate: f64,
    channels: usize,
}

impl Mic {
    /// Start recording from the default microphone.
    pub fn start() -> Result<Mic, String> {
        let device = cpal::default_host().default_input_device().ok_or("no microphone found")?;
        let config = device.default_input_config().map_err(|e| e.to_string())?;
        let (rate, channels) = (config.sample_rate() as f64, config.channels() as usize);
        let samples = Arc::new(Mutex::new(Vec::new()));

        let stream = match config.sample_format() {
            SampleFormat::F32 => record::<f32>(&device, config.into(), samples.clone()),
            SampleFormat::I16 => record::<i16>(&device, config.into(), samples.clone()),
            SampleFormat::U16 => record::<u16>(&device, config.into(), samples.clone()),
            SampleFormat::I32 => record::<i32>(&device, config.into(), samples.clone()),
            other => return Err(format!("unsupported microphone format {other:?}")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Mic { stream, samples, rate, channels })
    }

    /// Stop recording and return the audio as 16 kHz mono samples in [-1, 1]
    /// (Whisper's format), converted from whatever the microphone gives.
    pub fn stop(self) -> Vec<f32> {
        drop(self.stream);
        let raw = std::mem::take(&mut *self.samples.lock().unwrap());
        // Mix the channels down to one (mono) by averaging each frame.
        let mono: Vec<f32> = raw.chunks(self.channels.max(1)).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect();
        resample(&mono, self.rate, WHISPER_RATE)
    }
}

/// Open an input stream that appends every sample (as f32) to `samples`.
fn record<T>(device: &cpal::Device, config: cpal::StreamConfig, samples: Arc<Mutex<Vec<f32>>>) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &_| {
                if let Ok(mut buf) = samples.lock() {
                    buf.extend(data.iter().map(|s| s.to_sample::<f32>()));
                }
            },
            |err| eprintln!("microphone error: {err}"),
            None,
        )
        .map_err(|e| e.to_string())
}

/// Change the sample rate by straight-line interpolation between neighbouring samples.
/// Good enough for speech; Whisper doesn't need studio quality.
fn resample(input: &[f32], from: f64, to: f64) -> Vec<f32> {
    if input.is_empty() || from == to {
        return input.to_vec();
    }
    let out_len = (input.len() as f64 * to / from) as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * from / to;
            let j = pos as usize;
            let frac = (pos - j as f64) as f32;
            let a = input[j.min(input.len() - 1)];
            let b = input[(j + 1).min(input.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

/// Speaks replies aloud with Windows' built-in voice.
pub struct Voice {
    tts: tts::Tts,
}

impl Voice {
    pub fn new() -> Result<Voice, String> {
        Ok(Voice { tts: tts::Tts::default().map_err(|e| e.to_string())? })
    }

    /// Say `text` aloud, interrupting anything still being said. Returns immediately.
    pub fn say(&mut self, text: &str) {
        let _ = self.tts.speak(text, true);
    }

    /// Stop talking now.
    pub fn stop(&mut self) {
        let _ = self.tts.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_keeps_a_tone_and_changes_the_length() {
        // 0.1 s of a 440 Hz tone at 48 kHz → 0.1 s at 16 kHz.
        let tone: Vec<f32> = (0..4800).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin()).collect();
        let out = resample(&tone, 48_000.0, 16_000.0);
        assert_eq!(out.len(), 1600);
        for (i, s) in out.iter().enumerate() {
            let expected = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
            assert!((s - expected).abs() < 0.01);
        }
    }
}
