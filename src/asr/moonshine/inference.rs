use ort::value::TensorRef;

use super::session::*;

pub(super) fn process_frontend(
    sessions: &mut MoonshineSessions,
    config: &StreamingConfig,
    state: &mut StreamingState,
    audio: &[f32],
) {
    let audio_shape = [1i64, audio.len() as i64];
    let sb_shape = [1i64, 79];
    let sl_shape = [1i64];
    let sl_data = [state.sample_len];
    let c1_shape = [1i64, config.d_model_frontend, 4];
    let c2_shape = [1i64, config.c1, 4];
    let fc_shape = [1i64];
    let fc_data = [state.frame_count];
    let outputs = sessions.frontend.run(ort::inputs![
        "audio_chunk" => TensorRef::from_array_view((&audio_shape[..], audio)).expect("audio tensor"),
        "sample_buffer" => TensorRef::from_array_view((&sb_shape[..], &state.sample_buffer[..])).expect("sb tensor"),
        "sample_len" => TensorRef::from_array_view((&sl_shape[..], &sl_data[..])).expect("sl tensor"),
        "conv1_buffer" => TensorRef::from_array_view((&c1_shape[..], &state.conv1_buffer[..])).expect("c1 tensor"),
        "conv2_buffer" => TensorRef::from_array_view((&c2_shape[..], &state.conv2_buffer[..])).expect("c2 tensor"),
        "frame_count" => TensorRef::from_array_view((&fc_shape[..], &fc_data[..])).expect("fc tensor"),
    ]).expect("frontend inference failed");
    let (feat_shape, feat_data) = outputs["features"]
        .try_extract_tensor::<f32>()
        .expect("extract features");
    let num_features = feat_shape[1] as i32;
    let feat_dim = feat_shape[2] as usize;
    if num_features > 0 {
        let feat_count = num_features as usize * feat_dim;
        state
            .accumulated_features
            .extend_from_slice(&feat_data[..feat_count]);
        state.accumulated_feature_count += num_features;
    }
    let (_, sb_data) = outputs["sample_buffer_out"]
        .try_extract_tensor::<f32>()
        .expect("extract sb_out");
    state.sample_buffer.copy_from_slice(sb_data);
    let (_, sl_data) = outputs["sample_len_out"]
        .try_extract_tensor::<i64>()
        .expect("extract sl_out");
    state.sample_len = sl_data[0];
    let (_, c1_data) = outputs["conv1_buffer_out"]
        .try_extract_tensor::<f32>()
        .expect("extract c1_out");
    state.conv1_buffer.copy_from_slice(c1_data);
    let (_, c2_data) = outputs["conv2_buffer_out"]
        .try_extract_tensor::<f32>()
        .expect("extract c2_out");
    state.conv2_buffer.copy_from_slice(c2_data);
    let (_, fc_data) = outputs["frame_count_out"]
        .try_extract_tensor::<i64>()
        .expect("extract fc_out");
    state.frame_count = fc_data[0];
}

pub(super) fn run_encoder(
    sessions: &mut MoonshineSessions,
    config: &StreamingConfig,
    state: &mut StreamingState,
    is_final: bool,
) {
    let total_features = state.accumulated_feature_count;
    if total_features == 0 {
        return;
    }
    let mut stable_count = (total_features - config.total_lookahead as i32).max(0);
    if is_final {
        stable_count = total_features;
    }
    let new_frames = stable_count - state.encoder_frames_emitted;
    if new_frames <= 0 {
        return;
    }
    let left_context = (16 * config.depth) as i32;
    let window_start = (state.encoder_frames_emitted - left_context).max(0);
    let window_size = total_features - window_start;
    let enc_dim = config.encoder_dim as usize;
    let feat_offset = window_start as usize * enc_dim;
    let feat_len = window_size as usize * enc_dim;
    let features_slice = &state.accumulated_features[feat_offset..feat_offset + feat_len];
    let feat_shape = [1i64, window_size as i64, config.encoder_dim];
    let enc_outputs = sessions.encoder.run(ort::inputs![
        "features" => TensorRef::from_array_view((&feat_shape[..], features_slice)).expect("enc feat tensor"),
    ]).expect("encoder inference failed");
    let (enc_shape, encoded_data) = enc_outputs["encoded"]
        .try_extract_tensor::<f32>()
        .expect("extract encoded");
    let total_encoded = enc_shape[1] as i32;
    let start_idx = state.encoder_frames_emitted - window_start;
    assert!(
        start_idx >= 0 && start_idx + new_frames <= total_encoded,
        "encoder window misaligned: start={start_idx}, new={new_frames}, total={total_encoded}"
    );
    let new_encoded_offset = start_idx as usize * enc_dim;
    let new_encoded_len = new_frames as usize * enc_dim;
    let new_encoded = &encoded_data[new_encoded_offset..new_encoded_offset + new_encoded_len];
    let enc_slice_shape = [1i64, new_frames as i64, config.encoder_dim];
    let pos_shape = [1i64];
    let pos_data = [state.adapter_pos_offset];
    let adapter_outputs = sessions.adapter.run(ort::inputs![
        "encoded" => TensorRef::from_array_view((&enc_slice_shape[..], new_encoded)).expect("adapter enc tensor"),
        "pos_offset" => TensorRef::from_array_view((&pos_shape[..], &pos_data[..])).expect("adapter pos tensor"),
    ]).expect("adapter inference failed");
    let (_, mem_data) = adapter_outputs["memory"]
        .try_extract_tensor::<f32>()
        .expect("extract adapter memory");
    let mem_count = new_frames as usize * config.decoder_dim as usize;
    state.memory.extend_from_slice(&mem_data[..mem_count]);
    state.memory_len += new_frames;
    state.cross_kv_valid = false;
    state.encoder_frames_emitted = stable_count;
    state.adapter_pos_offset += new_frames as i64;
}
