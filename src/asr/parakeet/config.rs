use crate::platform::coreml;
use crate::settings::ModelId;
use objc2::rc::Retained;
use objc2_core_ml::{MLComputeUnits, MLModel, MLMultiArray, MLMultiArrayDataType};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub(super) const DURATION_BINS: [usize; 5] = [0, 1, 2, 3, 4];
pub(super) const BUNDLE_NAMES: [&str; 4] = [
    "preprocessor.mlmodelc",
    "encoder.mlmodelc",
    "decoder.mlmodelc",
    "joint_decision.mlmodelc",
];

#[derive(Clone, Copy)]
#[rustfmt::skip]
pub(super) struct ParakeetConfig {
    pub encoder_hidden: usize, pub decoder_hidden: usize, pub decoder_layers: usize,
    pub max_audio_samples: usize, pub min_audio_samples: usize, pub max_encoder_frames: usize,
    pub blank_id: i32, pub encoder_output_name: &'static str, pub encoder_length_name: &'static str,
    pub encoder_mel_input_name: &'static str, pub encoder_length_input_name: &'static str,
}

impl ParakeetConfig {
    fn shared_base() -> Self {
        Self {
            encoder_hidden: 1024,
            decoder_hidden: 640,
            decoder_layers: 2,
            max_audio_samples: 240_000,
            min_audio_samples: 16_000,
            max_encoder_frames: 188,
            blank_id: 0,
            encoder_output_name: "",
            encoder_length_name: "",
            encoder_mel_input_name: "",
            encoder_length_input_name: "",
        }
    }
    pub(super) fn for_model(id: ModelId) -> Self {
        match id {
            ModelId::Multilingual => Self {
                blank_id: 8192,
                encoder_output_name: "encoder_output",
                encoder_length_name: "encoder_output_length",
                encoder_mel_input_name: "audio_signal",
                encoder_length_input_name: "length",
                ..Self::shared_base()
            },
            ModelId::English => Self {
                blank_id: 1024,
                encoder_output_name: "encoder",
                encoder_length_name: "encoder_length",
                encoder_mel_input_name: "mel",
                encoder_length_input_name: "mel_length",
                ..Self::shared_base()
            },
            _ => unreachable!(),
        }
    }
}

struct UnsafeSend<T>(T);
unsafe impl<T> Send for UnsafeSend<T> {}

#[rustfmt::skip]
pub(super) struct AsrModels { pub(super) preprocessor: Retained<MLModel>, pub(super) encoder: Retained<MLModel>, pub(super) decoder: Retained<MLModel>, pub(super) joint: Retained<MLModel> }

#[rustfmt::skip]
pub(super) fn load_models(model_dir: &Path) -> AsrModels {
    let (preprocessor, encoder, decoder, joint) = std::thread::scope(|s| {
        let mut it = BUNDLE_NAMES.iter().map(|name| s.spawn(move || UnsafeSend(coreml::load_model(&model_dir.join(name), MLComputeUnits::All)))).collect::<Vec<_>>().into_iter().map(|h| h.join().unwrap().0);
        (it.next().unwrap(), it.next().unwrap(), it.next().unwrap(), it.next().unwrap())
    });
    AsrModels { preprocessor, encoder, decoder, joint }
}

#[rustfmt::skip]
pub(super) struct AsrBuffers {
    pub(super) h_state: Retained<MLMultiArray>, pub(super) c_state: Retained<MLMultiArray>, pub(super) pre_audio_input: Retained<MLMultiArray>,
    pub(super) pre_length_input: Retained<MLMultiArray>, pub(super) enc_length_input: Retained<MLMultiArray>, pub(super) dec_targets: Retained<MLMultiArray>,
    pub(super) dec_target_length: Retained<MLMultiArray>, pub(super) joint_encoder_step: Retained<MLMultiArray>,
}

#[rustfmt::skip]
pub(super) fn create_buffers(cfg: &ParakeetConfig) -> AsrBuffers {
    let h_state = coreml::create_multi_array(&[cfg.decoder_layers, 1, cfg.decoder_hidden], MLMultiArrayDataType::Float32);
    let c_state = coreml::create_multi_array(&[cfg.decoder_layers, 1, cfg.decoder_hidden], MLMultiArrayDataType::Float32);
    let pre_audio_input = coreml::create_multi_array(&[1, cfg.max_audio_samples], MLMultiArrayDataType::Float32);
    let pre_length_input = coreml::create_multi_array(&[1], MLMultiArrayDataType::Int32);
    let enc_length_input = coreml::create_multi_array(&[1], MLMultiArrayDataType::Int32);
    let dec_targets = coreml::create_multi_array(&[1, 1], MLMultiArrayDataType::Int32);
    let dec_target_length = coreml::create_multi_array(&[1], MLMultiArrayDataType::Int32);
    unsafe { *coreml::multi_array_i32_ptr(&dec_target_length) = 1 };
    let joint_encoder_step = coreml::create_multi_array(&[1, cfg.encoder_hidden, 1], MLMultiArrayDataType::Float32);
    AsrBuffers { h_state, c_state, pre_audio_input, pre_length_input, enc_length_input, dec_targets, dec_target_length, joint_encoder_step }
}

#[rustfmt::skip]
pub(super) struct Vocab { pub(super) tokens: HashMap<u32, String> }

#[rustfmt::skip]
#[allow(clippy::manual_strip)]
impl Vocab {
    pub(super) fn load(path: &Path) -> Self {
        let data = std::fs::read_to_string(path).expect("failed to read vocab file");
        let json: Value = serde_json::from_str(&data).expect("invalid vocab JSON");
        let obj = json.as_object().expect("vocab must be a JSON object");
        let mut tokens = HashMap::with_capacity(obj.len());
        for (key, value) in obj {
            let id: u32 = key.parse().expect("vocab key must be numeric");
            let token = value.as_str().expect("vocab value must be string").to_string();
            tokens.insert(id, token);
        }
        Self { tokens }
    }
    pub(super) fn decode(&self, token_ids: &[i32]) -> String {
        token_ids.iter().filter_map(|&id| self.tokens.get(&(id as u32))).fold(String::with_capacity(token_ids.len() * 4), |mut text, token| {
            if token.starts_with('\u{2581}') {
                text.push(' ');
                text.push_str(&token[3..]);
            }
            if !token.starts_with('\u{2581}') {
                text.push_str(token);
            }
            text
        }).trim().to_string()
    }
}
