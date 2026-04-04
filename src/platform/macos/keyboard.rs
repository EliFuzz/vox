use crate::audio::AudioCapture;
use crate::recording::{start_recording, stop_recording};
use crate::tray::TrayMsg;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DOUBLE_TAP_WINDOW_MS: u128 = 500;

pub fn listen(
    is_recording: Arc<AtomicBool>,
    recording_mode: Arc<AtomicU8>,
    audio: Arc<AudioCapture>,
    models_ready: Arc<AtomicBool>,
    should_quit: Arc<AtomicBool>,
    tx: std::sync::mpsc::Sender<TrayMsg>,
) {
    let key = crate::settings::load()
        .shortcut
        .unwrap_or(rdev::Key::AltGr);
    let mut key_held = false;
    let mut last_key_tap: Option<Instant> = None;
    if let Err(e) = rdev::listen(move |event| {
        if should_quit.load(Ordering::Acquire) {
            return;
        }
        match event.event_type {
            rdev::EventType::KeyPress(pressed) if pressed == key => {
                if key_held {
                    return;
                }
                key_held = true;
                if !models_ready.load(Ordering::Acquire) {
                    return;
                }
                let now = Instant::now();
                let is_double_tap = last_key_tap
                    .map(|t| now.duration_since(t).as_millis() < DOUBLE_TAP_WINDOW_MS)
                    .unwrap_or(false);
                last_key_tap = Some(now);
                if is_double_tap {
                    let was_recording = is_recording.load(Ordering::Acquire);
                    if was_recording && recording_mode.load(Ordering::Acquire) == crate::MODE_TTT {
                        stop_recording(&is_recording, &audio, &tx);
                    }
                    if !was_recording {
                        start_recording(
                            &is_recording,
                            &recording_mode,
                            &audio,
                            &tx,
                            crate::MODE_TTT,
                        );
                    }
                    return;
                }
                if !is_recording.load(Ordering::Acquire) {
                    start_recording(&is_recording, &recording_mode, &audio, &tx, crate::MODE_PTT);
                }
            }
            rdev::EventType::KeyRelease(released) if released == key => {
                key_held = false;
                if is_recording.load(Ordering::Acquire)
                    && recording_mode.load(Ordering::Acquire) == crate::MODE_PTT
                {
                    stop_recording(&is_recording, &audio, &tx);
                }
            }
            _ => {}
        }
    }) {
        eprintln!(
            "keyboard listener failed: {e:?} - ensure Accessibility/Input Monitoring permissions"
        );
        std::process::exit(1);
    }
}
