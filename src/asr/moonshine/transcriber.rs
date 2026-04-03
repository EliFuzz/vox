use crate::asr::engine::AsrEngine;
use std::path::Path;

use super::decoder::decode_full;
use super::inference::{process_frontend, run_encoder};
use super::session::*;
use super::tokenizer::BinTokenizer;

pub struct MoonshineAsr {
    sessions: MoonshineSessions,
    config: StreamingConfig,
    state: StreamingState,
    tokenizer: BinTokenizer,
}

unsafe impl Send for MoonshineAsr {}

impl MoonshineAsr {
    pub fn new(model_dir: &Path) -> Self {
        let cfg = load_streaming_config(model_dir);
        let sessions = load_sessions(model_dir);
        let state = StreamingState::new(&cfg);
        let tokenizer = BinTokenizer::load(&model_dir.join("tokenizer.bin"));
        let mut asr = Self {
            sessions,
            config: cfg,
            state,
            tokenizer,
        };
        asr.prewarm();
        asr
    }

    fn prewarm(&mut self) {
        let dummy = vec![0.0f32; SAMPLE_RATE];
        let _ = self.transcribe_batch(&dummy);
        self.state.reset(&self.config);
    }

    fn transcribe_batch(&mut self, audio: &[f32]) -> String {
        self.state.reset(&self.config);
        process_frontend(&mut self.sessions, &self.config, &mut self.state, audio);
        run_encoder(&mut self.sessions, &self.config, &mut self.state, true);
        if self.state.memory_len == 0 {
            return String::new();
        }
        let tokens = decode_full(&mut self.sessions, &self.config, &mut self.state);
        self.tokenizer.decode(&tokens)
    }
}

impl AsrEngine for MoonshineAsr {
    fn max_chunk_samples(&self) -> usize {
        MAX_CHUNK_SAMPLES
    }

    fn transcribe_chunk(&mut self, audio: &[f32]) -> String {
        self.transcribe_batch(audio)
    }
}
