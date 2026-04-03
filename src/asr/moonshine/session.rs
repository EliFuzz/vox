use ort::session::Session;
use serde::Deserialize;
use std::path::Path;

pub(super) const SAMPLE_RATE: usize = 16_000;
pub(super) const MAX_CHUNK_SAMPLES: usize = SAMPLE_RATE * 30;
pub(super) const MAX_TOKENS_PER_SECOND: f32 = 6.5;
pub(super) const TYPICAL_FEATURES: usize = 256;
pub(super) const TYPICAL_MEMORY: usize = 256;
pub(super) const TYPICAL_KV: usize = 64 * 1024;

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub(super) struct StreamingConfig {
    pub encoder_dim: i64,
    pub decoder_dim: i64,
    pub depth: i64,
    pub nheads: i64,
    pub head_dim: i64,
    pub vocab_size: i64,
    pub bos_id: i64,
    pub eos_id: i64,
    pub total_lookahead: i64,
    pub d_model_frontend: i64,
    pub c1: i64,
    pub max_seq_len: i64,
}

pub(super) fn load_streaming_config(model_dir: &Path) -> StreamingConfig {
    let path = model_dir.join("streaming_config.json");
    let data = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "failed to read streaming_config.json from {}",
            model_dir.display()
        )
    });
    let mut cfg: StreamingConfig =
        serde_json::from_str(&data).expect("invalid streaming_config.json");
    if cfg.max_seq_len <= 0 {
        cfg.max_seq_len = 448;
    }
    cfg
}

pub(super) struct MoonshineSessions {
    pub(super) frontend: Session,
    pub(super) encoder: Session,
    pub(super) adapter: Session,
    pub(super) cross_kv: Session,
    pub(super) decoder_kv: Session,
}

fn build_session(path: &Path, intra_threads: usize) -> Session {
    Session::builder()
        .expect("failed to create session builder")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .expect("failed to set optimization level")
        .with_intra_threads(intra_threads)
        .expect("failed to set intra threads")
        .with_inter_threads(1)
        .expect("failed to set inter threads")
        .with_memory_pattern(true)
        .expect("failed to set memory pattern")
        .with_flush_to_zero()
        .expect("failed to set flush to zero")
        .with_intra_op_spinning(false)
        .expect("failed to set intra op spinning")
        .with_inter_op_spinning(false)
        .expect("failed to set inter op spinning")
        .commit_from_file(path)
        .unwrap_or_else(|e| panic!("failed to load ORT model {}: {e}", path.display()))
}

pub(super) fn load_sessions(model_dir: &Path) -> MoonshineSessions {
    let (frontend, encoder, adapter, cross_kv, decoder_kv) = std::thread::scope(|s| {
        let t1 = s.spawn(|| build_session(&model_dir.join("frontend.ort"), 1));
        let t2 = s.spawn(|| build_session(&model_dir.join("encoder.ort"), 2));
        let t3 = s.spawn(|| build_session(&model_dir.join("adapter.ort"), 1));
        let t4 = s.spawn(|| build_session(&model_dir.join("cross_kv.ort"), 1));
        let t5 = s.spawn(|| build_session(&model_dir.join("decoder_kv.ort"), 1));
        (
            t1.join().expect("frontend session load panicked"),
            t2.join().expect("encoder session load panicked"),
            t3.join().expect("adapter session load panicked"),
            t4.join().expect("cross_kv session load panicked"),
            t5.join().expect("decoder_kv session load panicked"),
        )
    });
    MoonshineSessions {
        frontend,
        encoder,
        adapter,
        cross_kv,
        decoder_kv,
    }
}

pub(super) struct StreamingState {
    pub(super) sample_buffer: Vec<f32>,
    pub(super) sample_len: i64,
    pub(super) conv1_buffer: Vec<f32>,
    pub(super) conv2_buffer: Vec<f32>,
    pub(super) frame_count: i64,
    pub(super) accumulated_features: Vec<f32>,
    pub(super) accumulated_feature_count: i32,
    pub(super) encoder_frames_emitted: i32,
    pub(super) adapter_pos_offset: i64,
    pub(super) memory: Vec<f32>,
    pub(super) memory_len: i32,
    pub(super) k_self: Vec<f32>,
    pub(super) v_self: Vec<f32>,
    pub(super) cache_seq_len: i64,
    pub(super) k_cross: Vec<f32>,
    pub(super) v_cross: Vec<f32>,
    pub(super) cross_len: i64,
    pub(super) cross_kv_valid: bool,
    pub(super) logits_buffer: Vec<f32>,
}

impl StreamingState {
    pub(super) fn new(cfg: &StreamingConfig) -> Self {
        let enc_dim = cfg.encoder_dim as usize;
        let dec_dim = cfg.decoder_dim as usize;
        Self {
            sample_buffer: vec![0.0; 79],
            sample_len: 0,
            conv1_buffer: vec![0.0; cfg.d_model_frontend as usize * 4],
            conv2_buffer: vec![0.0; cfg.c1 as usize * 4],
            frame_count: 0,
            accumulated_features: Vec::with_capacity(TYPICAL_FEATURES * enc_dim),
            accumulated_feature_count: 0,
            encoder_frames_emitted: 0,
            adapter_pos_offset: 0,
            memory: Vec::with_capacity(TYPICAL_MEMORY * dec_dim),
            memory_len: 0,
            k_self: Vec::with_capacity(TYPICAL_KV),
            v_self: Vec::with_capacity(TYPICAL_KV),
            cache_seq_len: 0,
            k_cross: Vec::with_capacity(TYPICAL_KV),
            v_cross: Vec::with_capacity(TYPICAL_KV),
            cross_len: 0,
            cross_kv_valid: false,
            logits_buffer: Vec::with_capacity(cfg.vocab_size as usize),
        }
    }

    pub(super) fn reset(&mut self, cfg: &StreamingConfig) {
        *self = Self::new(cfg);
    }
}
