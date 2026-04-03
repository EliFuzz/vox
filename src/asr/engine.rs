use crate::settings::ModelId;
use std::path::Path;

pub fn ready_check_files(id: ModelId) -> &'static [&'static str] {
    match id {
        ModelId::Multilingual => &["encoder.mlmodelc"],
        ModelId::English => &["Encoder.mlmodelc"],
        ModelId::EnglishSmall => &[
            "frontend.ort",
            "encoder.ort",
            "adapter.ort",
            "cross_kv.ort",
            "decoder_kv.ort",
            "tokenizer.bin",
            "streaming_config.json",
        ],
    }
}

pub trait AsrEngine: Send {
    fn max_chunk_samples(&self) -> usize;
    fn transcribe_chunk(&mut self, audio: &[f32]) -> String;
}

pub struct Asr {
    engine: Box<dyn AsrEngine>,
}

unsafe impl Sync for Asr {}

impl Asr {
    pub fn new(model_id: ModelId, model_dir: &Path) -> Self {
        let engine: Box<dyn AsrEngine> = match model_id {
            ModelId::Multilingual | ModelId::English => {
                Box::new(super::parakeet::ParakeetAsr::new(model_id, model_dir))
            }
            ModelId::EnglishSmall => Box::new(super::moonshine::MoonshineAsr::new(model_dir)),
        };
        Self { engine }
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> String {
        if audio.is_empty() {
            return String::new();
        }
        let max = self.engine.max_chunk_samples();
        if audio.len() <= max {
            return self.engine.transcribe_chunk(audio);
        }
        audio
            .chunks(max)
            .map(|c| self.engine.transcribe_chunk(c))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
}
