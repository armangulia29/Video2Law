//! Speech-to-text with Whisper (OpenAI's model), running in pure Rust via candle.
//!
//! How Whisper hears:
//! 1. Sound → "mel spectrogram": a picture of which pitches are loud at each moment
//!    (80 pitch bands × 100 moments per second), the way an ear splits sound.
//! 2. The encoder reads that picture once.
//! 3. The decoder writes the text one token (word piece) at a time, each time picking
//!    the most likely next token ("greedy" decoding), until it writes <|endoftext|>.

use std::path::Path;

use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self, Config, audio, model::Whisper};
use tokenizers::Tokenizer;

/// Longest answer we let Whisper write for one push-to-talk (about 150 words).
const MAX_TOKENS: usize = 224;

/// A loaded Whisper model.
pub struct Ears {
    model: Whisper,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,
    device: Device,
    start_tokens: Vec<u32>, // <|startoftranscript|> <|notimestamps|>
    end_token: u32,         // <|endoftext|>
    suppress: Tensor,       // −∞ for tokens Whisper must never write, 0 otherwise
}

impl Ears {
    /// Load Whisper from a folder holding model.safetensors, config.json and tokenizer.json.
    pub fn load(dir: &Path) -> Result<Ears, String> {
        let err = |e: &dyn std::fmt::Display| e.to_string();
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.display()))?,
        )
        .map_err(|e| err(&e))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| err(&e))?;
        let device = Device::cuda_if_available(0).map_err(|e| err(&e))?;
        // SAFETY: the weights file is memory-mapped read-only; we never modify it.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], whisper::DTYPE, &device) }
            .map_err(|e| err(&e))?;
        let model = Whisper::load(&vb, config.clone()).map_err(|e| err(&e))?;

        let id = |t: &str| tokenizer.token_to_id(t).ok_or(format!("token {t} missing"));
        let start_tokens = vec![id(whisper::SOT_TOKEN)?, id(whisper::NO_TIMESTAMPS_TOKEN)?];
        let end_token = id(whisper::EOT_TOKEN)?;
        let suppress: Vec<f32> = (0..config.vocab_size as u32)
            .map(|t| if config.suppress_tokens.contains(&t) { f32::NEG_INFINITY } else { 0.0 })
            .collect();
        let suppress = Tensor::new(suppress.as_slice(), &device).map_err(|e| err(&e))?;
        let mel_filters = mel_filters(config.num_mel_bins, whisper::N_FFT, whisper::SAMPLE_RATE as f32);

        Ok(Ears { model, tokenizer, mel_filters, device, start_tokens, end_token, suppress })
    }

    /// Turn 16 kHz mono audio into English text.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
        self.decode(samples).map_err(|e| e.to_string())
    }

    fn decode(&mut self, samples: &[f32]) -> candle_core::Result<String> {
        // 1. Mel spectrogram of (at most) the first 30 seconds.
        let samples = &samples[..samples.len().min(whisper::N_SAMPLES)];
        let n_mel = self.model.config.num_mel_bins;
        let mel = audio::pcm_to_mel(&self.model.config, samples, &self.mel_filters);
        let frames = mel.len() / n_mel;
        let mel = Tensor::from_vec(mel, (1, n_mel, frames), &self.device)?.narrow(2, 0, frames.min(whisper::N_FRAMES))?;

        // 2. Encoder: read the whole sound once.
        let audio_features = self.model.encoder.forward(&mel, true)?;

        // 3. Decoder: write the text, one most-likely token at a time.
        let mut tokens = self.start_tokens.clone();
        for step in 0..MAX_TOKENS {
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let hidden = self.model.decoder.forward(&input, &audio_features, step == 0)?;
            let last = hidden.i((.., tokens.len() - 1..))?;
            let logits = self.model.decoder.final_linear(&last)?.i(0)?.i(0)?;
            let logits = logits.broadcast_add(&self.suppress)?;
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            if next == self.end_token {
                break;
            }
            tokens.push(next);
        }
        self.model.reset_kv_cache();

        let text = self.tokenizer.decode(&tokens[self.start_tokens.len()..], true).map_err(candle_core::Error::msg)?;
        Ok(text.trim().to_string())
    }
}

/// The mel filter bank: `n_mel` triangles over the 0–8000 Hz range, spaced like the
/// human ear hears pitch (evenly below 1 kHz, logarithmically above). Same numbers as
/// librosa.filters.mel(sr=16000, n_fft=400, n_mels=80), which Whisper was trained with.
/// Layout: n_mel rows × (n_fft/2 + 1) columns.
fn mel_filters(n_mel: usize, n_fft: usize, sample_rate: f32) -> Vec<f32> {
    // Slaney's mel scale: linear up to 1000 Hz, logarithmic above.
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let log_step = (6.4f32).ln() / 27.0;
    let hz_to_mel = |hz: f32| if hz < min_log_hz { hz / f_sp } else { min_log_mel + (hz / min_log_hz).ln() / log_step };
    let mel_to_hz = |m: f32| if m < min_log_mel { m * f_sp } else { min_log_hz * ((m - min_log_mel) * log_step).exp() };

    // n_mel + 2 edge frequencies, evenly spaced on the mel scale.
    let top = hz_to_mel(sample_rate / 2.0);
    let edges: Vec<f32> = (0..n_mel + 2).map(|i| mel_to_hz(top * i as f32 / (n_mel + 1) as f32)).collect();

    let bins = n_fft / 2 + 1;
    let mut filters = vec![0.0; n_mel * bins];
    for m in 0..n_mel {
        let (lo, mid, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        let area_norm = 2.0 / (hi - lo); // Slaney normalisation: each triangle has the same area
        for k in 0..bins {
            let f = k as f32 * sample_rate / n_fft as f32;
            let rising = (f - lo) / (mid - lo);
            let falling = (hi - f) / (hi - mid);
            filters[m * bins + k] = rising.min(falling).max(0.0) * area_norm;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_filters_match_librosa() {
        // Spot values from librosa.filters.mel(sr=16000, n_fft=400, n_mels=80).
        let f = mel_filters(80, 400, 16_000.0);
        assert_eq!(f.len(), 80 * 201);
        assert!((f[1] - 0.02486259).abs() < 1e-5, "filter 0, bin 1 = {}", f[1]);
        // Every filter has a peak and the bank covers the whole range.
        for m in 0..80 {
            assert!(f[m * 201..(m + 1) * 201].iter().any(|&v| v > 0.0), "filter {m} empty");
        }
    }
}
