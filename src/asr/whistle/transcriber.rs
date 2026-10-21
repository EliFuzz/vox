use crate::asr::engine::{AsrEngine, MAX_CHUNK_SAMPLES};
use crate::asr::whistle::needle::{load, transcribe};
use crate::settings;
use serde_json::{from_slice, Value};
use std::ffi::CString;
use std::fs::read;
use std::path::Path;

const OUTPUT_CAPACITY_BYTES: usize = 30 * 4096;

pub struct WhistleAsr {
    output_buffer: Vec<u8>,
    language: Option<CString>,
}

impl WhistleAsr {
    pub fn new(model_dir: &Path) -> Self {
        let model_bytes = read(model_dir.join("whistle.cact")).expect("read whistle.cact");
        load(&model_bytes).expect("whistle load failed");
        Self {
            output_buffer: vec![0; OUTPUT_CAPACITY_BYTES],
            language: CString::new(settings::load().language.trim().to_lowercase())
                .ok()
                .filter(|code| !code.as_bytes().is_empty()),
        }
    }

    fn parse_text(&self) -> String {
        let text_length = self
            .output_buffer
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(self.output_buffer.len());
        from_slice::<Value>(&self.output_buffer[..text_length])
            .ok()
            .and_then(|response| {
                response["text"]
                    .as_str()
                    .map(|text| text.trim().to_string())
            })
            .unwrap_or_default()
    }
}

impl AsrEngine for WhistleAsr {
    fn max_chunk_samples(&self) -> usize {
        MAX_CHUNK_SAMPLES
    }

    fn transcribe_chunk(&mut self, audio: &[f32]) -> String {
        if let Err(error) = transcribe(audio, self.language.as_ref(), &mut self.output_buffer) {
            eprintln!("whistle transcribe failed: {error}");
            return String::new();
        }
        self.parse_text()
    }
}
