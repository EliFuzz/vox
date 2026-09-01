use crate::audio::AudioCapture;
use crate::recording::{start_recording, stop_recording};
use crate::tray::TrayMsg;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DOUBLE_TAP_WINDOW_MS: u128 = 500;

pub(crate) fn all_shortcut_keys_held(held: &HashSet<rdev::Key>, shortcut: &[rdev::Key]) -> bool {
    shortcut.iter().all(|k| held.contains(k))
}

pub fn listen(
    is_recording: Arc<AtomicBool>,
    recording_mode: Arc<AtomicU8>,
    audio: Arc<AudioCapture>,
    models_ready: Arc<AtomicBool>,
    should_quit: Arc<AtomicBool>,
    tx: std::sync::mpsc::Sender<TrayMsg>,
) {
    let shortcut = crate::settings::resolve_shortcut(&crate::settings::load().shortcut);
    let mut held_keys: HashSet<rdev::Key> = HashSet::with_capacity(shortcut.len());
    let mut combo_active = false;
    let mut last_combo_tap: Option<Instant> = None;
    if let Err(e) = rdev::listen(move |event| {
        if should_quit.load(Ordering::Acquire) {
            return;
        }
        match event.event_type {
            rdev::EventType::KeyPress(pressed) if shortcut.contains(&pressed) => {
                held_keys.insert(pressed);
                if !all_shortcut_keys_held(&held_keys, &shortcut) || combo_active {
                    return;
                }
                combo_active = true;
                if !models_ready.load(Ordering::Acquire) {
                    return;
                }
                let now = Instant::now();
                let is_double_tap = last_combo_tap
                    .map(|t| now.duration_since(t).as_millis() < DOUBLE_TAP_WINDOW_MS)
                    .unwrap_or(false);
                last_combo_tap = Some(now);
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
            rdev::EventType::KeyRelease(released) if shortcut.contains(&released) => {
                held_keys.remove(&released);
                if combo_active {
                    combo_active = false;
                    if is_recording.load(Ordering::Acquire)
                        && recording_mode.load(Ordering::Acquire) == crate::MODE_PTT
                    {
                        stop_recording(&is_recording, &audio, &tx);
                    }
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
