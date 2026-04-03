use crate::platform::coreml;
use objc2::rc::Retained;
use objc2_core_ml::{MLComputeUnits, MLModel, MLMultiArray, MLMultiArrayDataType};
use std::path::Path;
const CHUNK_SAMPLES: usize = 4096;
const CONTEXT_SAMPLES: usize = 64;
const INPUT_SAMPLES: usize = CONTEXT_SAMPLES + CHUNK_SAMPLES;
const HIDDEN_SIZE: usize = 128;
pub struct Vad {
    model: Retained<MLModel>,
    h_state: Retained<MLMultiArray>,
    c_state: Retained<MLMultiArray>,
    audio_input: Retained<MLMultiArray>,
    context: [f32; CONTEXT_SAMPLES],
    speech_threshold: f32,
    silence_threshold: f32,
}
unsafe impl Send for Vad {}
unsafe impl Sync for Vad {}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadStatus {
    Speech,
    Silence,
    Unknown,
}
unsafe fn zero_states(h: &Retained<MLMultiArray>, c: &Retained<MLMultiArray>) {
    std::ptr::write_bytes(coreml::multi_array_f32_ptr(h), 0, HIDDEN_SIZE);
    std::ptr::write_bytes(coreml::multi_array_f32_ptr(c), 0, HIDDEN_SIZE);
}
unsafe fn fill_process_audio(ptr: *mut f32, context: &[f32; CONTEXT_SAMPLES], samples: &[f32]) {
    std::ptr::copy_nonoverlapping(context.as_ptr(), ptr, CONTEXT_SAMPLES);
    let copy_len = samples.len().min(CHUNK_SAMPLES);
    std::ptr::copy_nonoverlapping(samples.as_ptr(), ptr.add(CONTEXT_SAMPLES), copy_len);
    if copy_len < CHUNK_SAMPLES {
        std::slice::from_raw_parts_mut(
            ptr.add(CONTEXT_SAMPLES + copy_len),
            CHUNK_SAMPLES - copy_len,
        )
        .fill(0.0);
    }
}
unsafe fn copy_states_and_prob(
    new_h: &Retained<MLMultiArray>,
    new_c: &Retained<MLMultiArray>,
    prob_array: &Retained<MLMultiArray>,
    h_state: &Retained<MLMultiArray>,
    c_state: &Retained<MLMultiArray>,
) -> f32 {
    std::ptr::copy_nonoverlapping(
        coreml::multi_array_f32_ptr(new_h),
        coreml::multi_array_f32_ptr(h_state),
        HIDDEN_SIZE,
    );
    std::ptr::copy_nonoverlapping(
        coreml::multi_array_f32_ptr(new_c),
        coreml::multi_array_f32_ptr(c_state),
        HIDDEN_SIZE,
    );
    *coreml::multi_array_f32_ptr(prob_array)
}
fn update_context(context: &mut [f32; CONTEXT_SAMPLES], samples: &[f32]) {
    if samples.len() >= CONTEXT_SAMPLES {
        context.copy_from_slice(&samples[samples.len() - CONTEXT_SAMPLES..]);
        return;
    }
    context.fill(0.0);
    context[CONTEXT_SAMPLES - samples.len()..].copy_from_slice(samples);
}
fn classify_prob(prob: f32, speech_threshold: f32, silence_threshold: f32) -> VadStatus {
    if prob > speech_threshold {
        return VadStatus::Speech;
    }
    if prob < silence_threshold {
        return VadStatus::Silence;
    }
    VadStatus::Unknown
}
impl Vad {
    pub fn new(model_path: &Path, speech_threshold: f32) -> Self {
        let model = coreml::load_model(model_path, MLComputeUnits::All);
        let h_state = coreml::create_multi_array(&[1, HIDDEN_SIZE], MLMultiArrayDataType::Float32);
        let c_state = coreml::create_multi_array(&[1, HIDDEN_SIZE], MLMultiArrayDataType::Float32);
        let audio_input =
            coreml::create_multi_array(&[1, INPUT_SAMPLES], MLMultiArrayDataType::Float32);
        unsafe { zero_states(&h_state, &c_state) }
        Self {
            model,
            h_state,
            c_state,
            audio_input,
            context: [0.0; CONTEXT_SAMPLES],
            speech_threshold,
            silence_threshold: speech_threshold - 0.15,
        }
    }
    pub fn process(&mut self, samples: &[f32]) -> VadStatus {
        let ptr = coreml::multi_array_f32_ptr(&self.audio_input);
        unsafe { fill_process_audio(ptr, &self.context, samples) }
        update_context(&mut self.context, samples);
        let dict = coreml::make_input_dict(&[
            ("audio_input", &self.audio_input),
            ("hidden_state", &self.h_state),
            ("cell_state", &self.c_state),
        ]);
        let prob = coreml::predict_with(&self.model, &dict, None, |output| {
            let new_h = coreml::feature_value_multi_array(output, "new_hidden_state");
            let new_c = coreml::feature_value_multi_array(output, "new_cell_state");
            let prob_array = coreml::feature_value_multi_array(output, "vad_output");
            unsafe {
                copy_states_and_prob(&new_h, &new_c, &prob_array, &self.h_state, &self.c_state)
            }
        });
        classify_prob(prob, self.speech_threshold, self.silence_threshold)
    }
    pub fn reset(&mut self) {
        unsafe { zero_states(&self.h_state, &self.c_state) }
        self.context = [0.0; CONTEXT_SAMPLES];
    }
}
