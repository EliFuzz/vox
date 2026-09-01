use crate::asr::Asr;
use crate::audio::AudioCapture;
use crate::normalization::normalize_with_punctuation;
use crate::tray::{TrayMsg, TrayState};
use crate::vad::{Vad, VadStatus};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

const VAD_CHUNK_SAMPLES: usize = 4096;
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
    let mut samples = Vec::with_capacity(VAD_CHUNK_SAMPLES);
    let mut speech_buffer = Vec::with_capacity(STREAM_CHUNK_SAMPLES);
    let mut vad_buffer = Vec::with_capacity(VAD_CHUNK_SAMPLES * 2);
    let mut accumulated_text = Vec::new();
    let mut is_speech = false;
    let mut was_recording = false;
    loop {
        if should_quit.load(Ordering::Acquire) {
            break;
        }
        let currently_recording = is_recording.load(Ordering::Acquire);
        if was_recording && !currently_recording {
            was_recording = false;
            let dropped_samples = audio.dropped_samples();
            if dropped_samples > 0 {
                eprintln!("audio buffer overflow: dropped {dropped_samples} samples");
            }
            loop {
                audio.drain_into(VAD_CHUNK_SAMPLES * 4, &mut samples);
                if samples.is_empty() {
                    break;
                }
                if is_speech {
                    speech_buffer.extend_from_slice(&samples);
                }
            }
            if is_speech {
                transcribe(&speech_buffer, &asr_holder, &mut accumulated_text);
            }
            flush_text(&mut accumulated_text);
            if let Some(vad) = vad_holder.get() {
                if let Ok(mut vad) = vad.lock() {
                    vad.reset();
                }
            }
            is_speech = false;
            speech_buffer.clear();
            vad_buffer.clear();
            tx.send(TrayMsg::SetState(TrayState::Off)).ok();
            continue;
        }
        if !currently_recording {
            was_recording = false;
            wait_for_data(&data_signal, 100);
            continue;
        }
        was_recording = true;
        if !models_ready.load(Ordering::Acquire) {
            is_recording.store(false, Ordering::Release);
            audio.stop();
            continue;
        }
        let Some(vad) = vad_holder.get() else {
            models_ready.store(false, Ordering::Release);
            is_recording.store(false, Ordering::Release);
            audio.stop();
            continue;
        };
        audio.drain_into(VAD_CHUNK_SAMPLES, &mut samples);
        if samples.is_empty() {
            wait_for_data(&data_signal, 5);
            continue;
        }
        speech_buffer.extend_from_slice(&samples);
        vad_buffer.extend_from_slice(&samples);
        let mut consumed = 0;
        while vad_buffer.len() - consumed >= VAD_CHUNK_SAMPLES {
            let chunk = &vad_buffer[consumed..consumed + VAD_CHUNK_SAMPLES];
            let status = vad.lock().unwrap().process(chunk);
            consumed += VAD_CHUNK_SAMPLES;
            if status == VadStatus::Speech {
                is_speech = true;
                continue;
            }
            if !is_speech {
                let excess = speech_buffer.len().saturating_sub(VAD_CHUNK_SAMPLES);
                speech_buffer.drain(..excess);
                continue;
            }
            transcribe(&speech_buffer, &asr_holder, &mut accumulated_text);
            speech_buffer.clear();
            flush_text(&mut accumulated_text);
            vad.lock().unwrap().reset();
            is_speech = false;
        }
        if consumed > 0 {
            vad_buffer.drain(..consumed);
        }
        while is_speech && speech_buffer.len() >= STREAM_CHUNK_SAMPLES {
            let chunk: Vec<f32> = speech_buffer.drain(..STREAM_CHUNK_SAMPLES).collect();
            transcribe(&chunk, &asr_holder, &mut accumulated_text);
        }
    }
}

fn wait_for_data(signal: &(Mutex<()>, Condvar), timeout_ms: u64) {
    let _ = signal
        .1
        .wait_timeout(signal.0.lock().unwrap(), Duration::from_millis(timeout_ms))
        .unwrap();
}

fn transcribe(
    audio: &[f32],
    asr_holder: &Arc<Mutex<Option<Asr>>>,
    accumulated_text: &mut Vec<String>,
) {
    if let Some(asr) = asr_holder.lock().unwrap().as_mut() {
        let text = asr.transcribe(audio);
        if !text.is_empty() {
            accumulated_text.push(text);
        }
    }
}

fn flush_text(accumulated: &mut Vec<String>) {
    if accumulated.is_empty() {
        return;
    }
    let text = normalize_with_punctuation(&accumulated.join(" "));
    accumulated.clear();
    std::thread::spawn(move || {
        static PASTE_LOCK: Mutex<()> = Mutex::new(());
        let _guard = PASTE_LOCK.lock().unwrap();
        crate::output::paste_at_cursor(&text);
    });
}
