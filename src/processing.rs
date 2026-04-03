use crate::asr::Asr;
use crate::audio::AudioCapture;
use crate::normalization::normalize_with_punctuation;
use crate::tray::{TrayMsg, TrayState};
use crate::vad::Vad;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

const VAD_SAMPLE_RATE: usize = 16000;
const VAD_CHUNK_SAMPLES: usize = 4096;
const MIN_SPEECH_SAMPLES: usize = VAD_CHUNK_SAMPLES;
const MIN_SILENCE_MS: u64 = 50;
const MIN_SILENCE_SAMPLES: usize = (VAD_SAMPLE_RATE as u64 * MIN_SILENCE_MS / 1000) as usize;
const STREAM_CHUNK_SAMPLES: usize = 240_000;

pub fn run(
    audio: Arc<AudioCapture>,
    vad_holder: Arc<OnceLock<Mutex<Vad>>>,
    asr_holder: Arc<Mutex<Option<Asr>>>,
    is_recording: Arc<AtomicBool>,
    models_ready: Arc<AtomicBool>,
    should_quit: Arc<AtomicBool>,
    tx: std::sync::mpsc::Sender<TrayMsg>,
) {
    let data_signal = audio.data_ready_signal();
    let mut speech_buffer = Vec::with_capacity(VAD_SAMPLE_RATE * 30);
    let mut vad_buffer = Vec::with_capacity(VAD_CHUNK_SAMPLES * 2);
    let mut accumulated_text = Vec::new();
    let mut is_speech = false;
    let mut silence_samples = 0usize;
    let mut speech_samples = 0usize;
    let mut was_recording = false;
    loop {
        if should_quit.load(Ordering::Acquire) {
            break;
        }
        let currently_recording = is_recording.load(Ordering::Acquire);
        if was_recording && !currently_recording {
            was_recording = false;
            let has_speech = is_speech && speech_samples >= MIN_SPEECH_SAMPLES;
            if !has_speech {
                while !audio.drain_chunk(VAD_CHUNK_SAMPLES * 4).is_empty() {}
            }
            if has_speech {
                loop {
                    let remaining = audio.drain_chunk(VAD_CHUNK_SAMPLES * 4);
                    if remaining.is_empty() {
                        break;
                    }
                    speech_buffer.extend_from_slice(&remaining);
                }
                if models_ready.load(Ordering::Acquire) {
                    let audio_data = std::mem::take(&mut speech_buffer);
                    if let Some(ref mut asr) = *asr_holder.lock().unwrap() {
                        let text = asr.transcribe(&audio_data);
                        if !text.is_empty() {
                            accumulated_text.push(text);
                        }
                    }
                }
            }
            flush_text(&mut accumulated_text);
            if let Some(vad) = vad_holder.get() {
                if let Ok(mut v) = vad.lock() {
                    v.reset();
                }
            }
            is_speech = false;
            silence_samples = 0;
            speech_samples = 0;
            speech_buffer.clear();
            vad_buffer.clear();
            tx.send(TrayMsg::SetState(TrayState::Off)).ok();
            continue;
        }
        if !currently_recording {
            was_recording = false;
            let (lock, cvar) = &*data_signal;
            let guard = lock.lock().unwrap();
            let _unused = cvar
                .wait_timeout(guard, Duration::from_millis(100))
                .unwrap();
            continue;
        }
        was_recording = true;
        if !models_ready.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        let samples = audio.drain_chunk(VAD_CHUNK_SAMPLES);
        if samples.is_empty() {
            let (lock, cvar) = &*data_signal;
            let guard = lock.lock().unwrap();
            let _unused = cvar.wait_timeout(guard, Duration::from_millis(5)).unwrap();
            continue;
        }
        speech_buffer.extend_from_slice(&samples);
        vad_buffer.extend_from_slice(&samples);
        while vad_buffer.len() >= VAD_CHUNK_SAMPLES {
            let chunk: Vec<f32> = vad_buffer.drain(..VAD_CHUNK_SAMPLES).collect();
            let status = vad_holder.get().unwrap().lock().unwrap().process(&chunk);
            if matches!(status, crate::vad::VadStatus::Speech) {
                is_speech = true;
                silence_samples = 0;
                speech_samples += VAD_CHUNK_SAMPLES;
                continue;
            }
            if !is_speech {
                continue;
            }
            silence_samples += VAD_CHUNK_SAMPLES;
            if silence_samples < MIN_SILENCE_SAMPLES {
                continue;
            }
            if speech_samples < MIN_SPEECH_SAMPLES && accumulated_text.is_empty() {
                continue;
            }
            if speech_buffer.len() >= MIN_SPEECH_SAMPLES {
                let audio_data = std::mem::take(&mut speech_buffer);
                if let Some(ref mut asr) = *asr_holder.lock().unwrap() {
                    let text = asr.transcribe(&audio_data);
                    if !text.is_empty() {
                        accumulated_text.push(text);
                    }
                }
            }
            flush_text(&mut accumulated_text);
            vad_holder.get().unwrap().lock().unwrap().reset();
            is_speech = false;
            silence_samples = 0;
            speech_samples = 0;
            speech_buffer = Vec::with_capacity(VAD_SAMPLE_RATE * 30);
        }
        while is_speech && speech_buffer.len() >= STREAM_CHUNK_SAMPLES {
            let chunk: Vec<f32> = speech_buffer.drain(..STREAM_CHUNK_SAMPLES).collect();
            if let Some(ref mut asr) = *asr_holder.lock().unwrap() {
                let text = asr.transcribe(&chunk);
                if !text.is_empty() {
                    accumulated_text.push(text);
                }
            }
        }
    }
}

fn flush_text(accumulated: &mut Vec<String>) {
    if accumulated.is_empty() {
        return;
    }
    let full_text = normalize_with_punctuation(&accumulated.join(" "));
    accumulated.clear();
    std::thread::spawn(move || {
        static PASTE_LOCK: Mutex<()> = Mutex::new(());
        let _guard = PASTE_LOCK.lock().unwrap();
        crate::output::paste_at_cursor(&full_text);
    });
}
