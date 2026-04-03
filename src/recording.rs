use crate::audio::AudioCapture;
use crate::tray::{TrayMsg, TrayState};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::Sender;

pub(crate) fn start_recording(
    is_recording: &AtomicBool,
    recording_mode: &AtomicU8,
    audio: &AudioCapture,
    tx: &Sender<TrayMsg>,
    mode: u8,
) {
    recording_mode.store(mode, Ordering::Release);
    audio.start();
    is_recording.store(true, Ordering::Release);
    tx.send(TrayMsg::SetState(TrayState::On)).ok();
}

pub(crate) fn stop_recording(
    is_recording: &AtomicBool,
    audio: &AudioCapture,
    tx: &Sender<TrayMsg>,
) {
    is_recording.store(false, Ordering::Release);
    audio.stop();
    tx.send(TrayMsg::SetState(TrayState::Off)).ok();
}
