use super::config::{
    create_buffers, load_models, AsrBuffers, AsrModels, ParakeetConfig, Vocab, DURATION_BINS,
};
use crate::asr::engine::AsrEngine;
use crate::platform::coreml;
use crate::settings::ModelId;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_core_ml::MLMultiArray;
use std::path::Path;

#[rustfmt::skip]
pub struct ParakeetAsr { models: AsrModels, vocab: Vocab, buffers: AsrBuffers, cfg: ParakeetConfig }

unsafe impl Send for ParakeetAsr {}

fn encoder_strides(encoder_hidden: usize, shape: &[isize], strides: &[isize]) -> (isize, isize) {
    if shape.len() != 3 {
        return (encoder_hidden as isize, 1);
    }
    let s1 = strides.get(1).copied().unwrap_or(1);
    let s2 = strides.get(2).copied().unwrap_or(1);
    if shape[1] == encoder_hidden as isize {
        return (s2, s1);
    }
    (s1, s2)
}

#[rustfmt::skip]
impl ParakeetAsr {
    pub fn new(model_id: ModelId, model_dir: &Path) -> Self {
        let cfg = ParakeetConfig::for_model(model_id);
        let models = load_models(model_dir);
        let vocab = Vocab::load(&model_dir.join("vocab.json"));
        let buffers = create_buffers(&cfg);
        let mut asr = Self { models, vocab, buffers, cfg };
        asr.prewarm();
        asr
    }
    fn prewarm(&mut self) {
        let dummy = vec![0.0f32; self.cfg.min_audio_samples];
        let _ = self.transcribe_chunk_inner(&dummy);
        self.reset_decoder_state();
    }
    fn transcribe_chunk_inner(&mut self, audio: &[f32]) -> String {
        self.reset_decoder_state();
        let pre = self.run_preprocessor(audio);
        let enc = self.run_encoder(&pre);
        let frames = self.get_encoder_length(&enc);
        if frames == 0 { return String::new(); }
        let tokens = self.tdt_decode(&enc, frames);
        self.vocab.decode(&tokens)
    }
    fn reset_decoder_state(&mut self) {
        let state_size = self.cfg.decoder_layers * self.cfg.decoder_hidden;
        unsafe {
            std::ptr::write_bytes(coreml::multi_array_f32_ptr(&self.buffers.h_state), 0, state_size);
            std::ptr::write_bytes(coreml::multi_array_f32_ptr(&self.buffers.c_state), 0, state_size);
        }
    }
    fn run_preprocessor(&self, audio: &[f32]) -> Retained<ProtocolObject<dyn objc2_core_ml::MLFeatureProvider>> {
        let ptr = coreml::multi_array_f32_ptr(&self.buffers.pre_audio_input);
        let copy_len = audio.len().min(self.cfg.max_audio_samples);
        unsafe {
            std::ptr::copy_nonoverlapping(audio.as_ptr(), ptr, copy_len);
            if copy_len < self.cfg.max_audio_samples {
                std::slice::from_raw_parts_mut(ptr.add(copy_len), self.cfg.max_audio_samples - copy_len).fill(0.0);
            }
        }
        unsafe { *coreml::multi_array_i32_ptr(&self.buffers.pre_length_input) = copy_len.max(self.cfg.min_audio_samples) as i32 };
        coreml::predict(&self.models.preprocessor, &coreml::make_input_dict(&[
            ("audio_signal", &self.buffers.pre_audio_input), ("audio_length", &self.buffers.pre_length_input),
        ]), None)
    }
    fn run_encoder(&self, pre: &ProtocolObject<dyn objc2_core_ml::MLFeatureProvider>) -> Retained<ProtocolObject<dyn objc2_core_ml::MLFeatureProvider>> {
        let mel = coreml::feature_value_multi_array(pre, "mel");
        let mel_len_arr = coreml::feature_value_multi_array(pre, "mel_length");
        let mel_len = unsafe { *coreml::multi_array_i32_ptr(&mel_len_arr) };
        unsafe { *coreml::multi_array_i32_ptr(&self.buffers.enc_length_input) = mel_len };
        coreml::predict(&self.models.encoder, &coreml::make_input_dict(&[
            (self.cfg.encoder_mel_input_name, &mel), (self.cfg.encoder_length_input_name, &self.buffers.enc_length_input),
        ]), None)
    }
    fn get_encoder_length(&self, enc: &ProtocolObject<dyn objc2_core_ml::MLFeatureProvider>) -> usize {
        let arr = coreml::feature_value_multi_array(enc, self.cfg.encoder_length_name);
        let raw = unsafe { *coreml::multi_array_i32_ptr(&arr) };
        if raw <= 0 {
            return 0;
        }
        (raw as usize).min(self.cfg.max_encoder_frames)
    }
    fn copy_encoder_frame(&self, enc_ptr: *mut f32, time_stride: isize, hidden_stride: isize, frame: usize) {
        let step_ptr = coreml::multi_array_f32_ptr(&self.buffers.joint_encoder_step);
        let frame_start = (frame as isize * time_stride) as usize;
        if hidden_stride == 1 {
            unsafe { std::ptr::copy_nonoverlapping(enc_ptr.add(frame_start), step_ptr, self.cfg.encoder_hidden) };
            return;
        }
        for h in 0..self.cfg.encoder_hidden {
            unsafe { *step_ptr.add(h) = *enc_ptr.add(frame_start + (h as isize * hidden_stride) as usize) };
        }
    }
    fn tdt_decode(&mut self, encoder_output: &ProtocolObject<dyn objc2_core_ml::MLFeatureProvider>, encoder_frames: usize) -> Vec<i32> {
        let enc_array = coreml::feature_value_multi_array(encoder_output, self.cfg.encoder_output_name);
        let enc_ptr = coreml::multi_array_f32_ptr(&enc_array);
        let blank_id = self.cfg.blank_id;
        let strides = coreml::multi_array_strides(&enc_array);
        let shape = coreml::multi_array_shape(&enc_array);
        let (time_stride, hidden_stride) = encoder_strides(self.cfg.encoder_hidden, shape.as_slice(), strides.as_slice());
        let mut decoder_out = self.run_decoder(blank_id);
        let mut tokens: Vec<i32> = Vec::with_capacity(256);
        let mut t: usize = 0;
        let mut last_t: usize = usize::MAX;
        let mut same_frame_count: usize = 0;
        let resolve_duration = |bin: i32| DURATION_BINS[(bin as usize).min(DURATION_BINS.len() - 1)];
        while t < encoder_frames {
            if t != last_t {
                same_frame_count = 0;
                last_t = t;
            }
            if t == last_t {
                same_frame_count += 1;
                if same_frame_count >= 10 {
                    t += 1;
                    same_frame_count = 0;
                    continue;
                }
            }
            self.copy_encoder_frame(enc_ptr, time_stride, hidden_stride, t);
            let (token, _prob, duration_bin) = self.run_joint(&decoder_out);
            let mut is_blank = token == blank_id;
            let duration = resolve_duration(duration_bin);
            if is_blank && duration == 0 {
                t += 1;
            }
            if !is_blank || duration != 0 {
                t += duration;
            }
            while t < encoder_frames && is_blank {
                self.copy_encoder_frame(enc_ptr, time_stride, hidden_stride, t);
                let (inner_token, _inner_prob, inner_dur_bin) = self.run_joint(&decoder_out);
                is_blank = inner_token == blank_id;
                let inner_duration = resolve_duration(inner_dur_bin);
                if is_blank && inner_duration == 0 {
                    t += 1;
                }
                if !is_blank || inner_duration != 0 {
                    t += inner_duration;
                }
                if !is_blank { tokens.push(inner_token); decoder_out = self.run_decoder(inner_token); }
            }
            if !is_blank && token != blank_id { tokens.push(token); decoder_out = self.run_decoder(token); }
        }
        tokens
    }
    fn run_decoder(&mut self, token: i32) -> Retained<MLMultiArray> {
        unsafe { *coreml::multi_array_i32_ptr(&self.buffers.dec_targets) = token };
        let dict = coreml::make_input_dict(&[
            ("targets", &self.buffers.dec_targets), ("target_length", &self.buffers.dec_target_length),
            ("h_in", &self.buffers.h_state), ("c_in", &self.buffers.c_state),
        ]);
        let output = coreml::predict(&self.models.decoder, &dict, None);
        let state_size = self.cfg.decoder_layers * self.cfg.decoder_hidden;
        autoreleasepool(|_| {
            let h_out = coreml::feature_value_multi_array(&output, "h_out");
            let c_out = coreml::feature_value_multi_array(&output, "c_out");
            unsafe {
                std::ptr::copy_nonoverlapping(coreml::multi_array_f32_ptr(&h_out), coreml::multi_array_f32_ptr(&self.buffers.h_state), state_size);
                std::ptr::copy_nonoverlapping(coreml::multi_array_f32_ptr(&c_out), coreml::multi_array_f32_ptr(&self.buffers.c_state), state_size);
            }
        });
        coreml::feature_value_multi_array(&output, "decoder")
    }
    fn run_joint(&self, decoder_out: &MLMultiArray) -> (i32, f32, i32) {
        let dict = coreml::make_input_dict(&[("encoder_step", &self.buffers.joint_encoder_step), ("decoder_step", decoder_out)]);
        let output = coreml::predict(&self.models.joint, &dict, None);
        autoreleasepool(|_| {
            let token_array = coreml::feature_value_multi_array(&output, "token_id");
            let prob_array = coreml::feature_value_multi_array(&output, "token_prob");
            let dur_array = coreml::feature_value_multi_array(&output, "duration");
            let token = unsafe { *coreml::multi_array_i32_ptr(&token_array) };
            let prob = unsafe { *coreml::multi_array_f32_ptr(&prob_array) };
            let duration = unsafe { *coreml::multi_array_i32_ptr(&dur_array) };
            (token, prob, duration)
        })
    }
}

#[rustfmt::skip]
impl AsrEngine for ParakeetAsr {
    fn max_chunk_samples(&self) -> usize { self.cfg.max_audio_samples }
    fn transcribe_chunk(&mut self, audio: &[f32]) -> String { self.transcribe_chunk_inner(audio) }
}
