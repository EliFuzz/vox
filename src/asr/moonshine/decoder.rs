use ndarray::{ArrayViewD, IxDyn};
use ort::value::TensorRef;

use super::session::*;

fn compute_cross_kv(
    sessions: &mut MoonshineSessions,
    config: &StreamingConfig,
    state: &mut StreamingState,
) {
    if state.memory_len == 0 {
        return;
    }
    let mem_shape = [1i64, state.memory_len as i64, config.decoder_dim];
    let outputs = sessions.cross_kv.run(ort::inputs![
        "memory" => TensorRef::from_array_view((&mem_shape[..], &state.memory[..])).expect("cross_kv mem tensor"),
    ]).expect("cross_kv inference failed");
    let (k_shape, k_data) = outputs["k_cross"]
        .try_extract_tensor::<f32>()
        .expect("extract k_cross");
    let (_, v_data) = outputs["v_cross"]
        .try_extract_tensor::<f32>()
        .expect("extract v_cross");
    let total_cross = k_data.len();
    state.cross_len = k_shape[3] as i64;
    state.k_cross.clear();
    state.k_cross.extend_from_slice(&k_data[..total_cross]);
    state.v_cross.clear();
    state.v_cross.extend_from_slice(&v_data[..total_cross]);
    state.cross_kv_valid = true;
}

fn run_decoder_step(
    sessions: &mut MoonshineSessions,
    config: &StreamingConfig,
    state: &mut StreamingState,
    tokens: &[i64],
) -> usize {
    if !state.cross_kv_valid {
        compute_cross_kv(sessions, config, state);
    }
    let token_len = tokens.len() as i64;
    let token_shape = [1i64, token_len];
    let cache_len = state.cache_seq_len;
    let kv_self_size = (config.depth * config.nheads * cache_len * config.head_dim) as usize;
    if state.k_self.len() != kv_self_size {
        state.k_self.resize(kv_self_size, 0.0);
        state.v_self.resize(kv_self_size, 0.0);
    }
    let kv_dim = |seq: usize| {
        IxDyn(&[
            config.depth as usize,
            1,
            config.nheads as usize,
            seq,
            config.head_dim as usize,
        ])
    };
    let kv_self_dim = kv_dim(cache_len as usize);
    let k_self_view =
        ArrayViewD::from_shape(kv_self_dim.clone(), &state.k_self[..]).expect("k_self view");
    let v_self_view = ArrayViewD::from_shape(kv_self_dim, &state.v_self[..]).expect("v_self view");
    let kv_cross_dim = kv_dim(state.cross_len as usize);
    let k_cross_view =
        ArrayViewD::from_shape(kv_cross_dim.clone(), &state.k_cross[..]).expect("k_cross view");
    let v_cross_view =
        ArrayViewD::from_shape(kv_cross_dim, &state.v_cross[..]).expect("v_cross view");
    let outputs = sessions.decoder_kv.run(ort::inputs![
        "token" => TensorRef::from_array_view((&token_shape[..], tokens)).expect("decoder token tensor"),
        "k_self" => TensorRef::from_array_view(k_self_view).expect("k_self tensor"),
        "v_self" => TensorRef::from_array_view(v_self_view).expect("v_self tensor"),
        "out_k_cross" => TensorRef::from_array_view(k_cross_view).expect("k_cross tensor"),
        "out_v_cross" => TensorRef::from_array_view(v_cross_view).expect("v_cross tensor"),
    ]).expect("decoder_kv inference failed");
    let (_, logits_data) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .expect("extract logits");
    let vocab_size = config.vocab_size as usize;
    state.logits_buffer.clear();
    state
        .logits_buffer
        .extend_from_slice(&logits_data[..vocab_size]);
    let (k_shape, k_data) = outputs["out_k_self"]
        .try_extract_tensor::<f32>()
        .expect("extract out_k_self");
    let (_, v_data) = outputs["out_v_self"]
        .try_extract_tensor::<f32>()
        .expect("extract out_v_self");
    let new_kv_len = k_data.len();
    state.cache_seq_len = k_shape[3] as i64;
    state.k_self.clear();
    state.k_self.extend_from_slice(&k_data[..new_kv_len]);
    state.v_self.clear();
    state.v_self.extend_from_slice(&v_data[..new_kv_len]);
    vocab_size
}

fn argmax(logits: &[f32]) -> i64 {
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as i64
}

pub(super) fn decode_full(
    sessions: &mut MoonshineSessions,
    config: &StreamingConfig,
    state: &mut StreamingState,
) -> Vec<i64> {
    let duration_sec = state.memory_len as f32 * 0.020;
    let max_tokens = ((duration_sec * MAX_TOKENS_PER_SECOND).ceil() as i64).min(config.max_seq_len);
    state.k_self.clear();
    state.v_self.clear();
    state.cache_seq_len = 0;
    let vocab_size = run_decoder_step(sessions, config, state, &[config.bos_id]);
    let mut current_token = argmax(&state.logits_buffer[..vocab_size]);
    let mut result = Vec::with_capacity(max_tokens as usize);
    while current_token != config.eos_id && (result.len() as i64) < max_tokens {
        result.push(current_token);
        let vocab_size = run_decoder_step(sessions, config, state, &[current_token]);
        current_token = argmax(&state.logits_buffer[..vocab_size]);
    }
    result
}
