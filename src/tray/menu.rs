use crate::audio;
use crate::settings::ModelId;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use tray_icon::menu::{CheckMenuItem, MenuEvent, MenuId};
use tray_icon::{Icon, TrayIcon};

use super::{TrayMsg, TrayState};

pub(crate) fn handle_menu_event(
    event: MenuEvent,
    quit_id: &MenuId,
    ptt_id: &MenuId,
    ttt_id: &MenuId,
    ptt_item: &CheckMenuItem,
    ttt_item: &CheckMenuItem,
    recording_mode: &Arc<AtomicU8>,
    should_quit: &Arc<AtomicBool>,
    tray: &TrayIcon,
    icons: &[Icon; 3],
    model_ids: &[(MenuId, ModelId)],
    model_items: &[CheckMenuItem],
    asr_holder: Arc<Mutex<Option<crate::asr::Asr>>>,
    models_ready: Arc<AtomicBool>,
    load_cancel: &mut Arc<AtomicBool>,
    tray_tx: std::sync::mpsc::Sender<TrayMsg>,
    device_ids: &[(MenuId, String)],
    device_items: &[CheckMenuItem],
    devices: &[String],
    selected_device: &Arc<Mutex<Option<String>>>,
    audio_capture: &Arc<audio::AudioCapture>,
) -> bool {
    match &event.id {
        id if id == quit_id => {
            should_quit.store(true, Ordering::Release);
            true
        }
        id if id == ptt_id => {
            ptt_item.set_checked(true);
            ttt_item.set_checked(false);
            recording_mode.store(crate::MODE_PTT, Ordering::Release);
            false
        }
        id if id == ttt_id => {
            ttt_item.set_checked(true);
            ptt_item.set_checked(false);
            recording_mode.store(crate::MODE_TTT, Ordering::Release);
            false
        }
        _ => match model_ids
            .iter()
            .find(|(id, _)| id == &event.id)
            .map(|(_, id)| *id)
        {
            Some(model_id) => {
                for (mi, &avail_id) in model_items.iter().zip(ModelId::ALL.iter()) {
                    mi.set_checked(avail_id == model_id);
                }
                let asr_h = asr_holder.clone();
                let ready = models_ready.clone();
                let tx_clone = tray_tx.clone();
                tray.set_icon(Some(icons[TrayState::Loading as usize].clone()))
                    .ok();
                ready.store(false, Ordering::Release);
                load_cancel.store(true, Ordering::Release);
                drop(asr_h.lock().unwrap().take());
                let cancel = Arc::new(AtomicBool::new(false));
                *load_cancel = cancel.clone();
                std::thread::spawn(move || {
                    let mut cfg = crate::settings::load();
                    cfg.model = model_id;
                    crate::settings::save(&cfg);
                    match crate::settings::model_ready(model_id) {
                        true => {}
                        false => match crate::model_download::ensure_model(model_id) {
                            Ok(()) => {}
                            Err(e) => {
                                eprintln!("model download failed for {}: {e}", model_id.as_str());
                                tx_clone.send(TrayMsg::SetState(TrayState::Off)).ok();
                                return;
                            }
                        },
                    }
                    if cancel.load(Ordering::Acquire) {
                        tx_clone.send(TrayMsg::SetState(TrayState::Off)).ok();
                        return;
                    }
                    let dir = crate::settings::model_path(model_id);
                    let asr = crate::asr::Asr::new(model_id, &dir);
                    if cancel.load(Ordering::Acquire) {
                        drop(asr);
                        tx_clone.send(TrayMsg::SetState(TrayState::Off)).ok();
                        return;
                    }
                    *asr_h.lock().unwrap() = Some(asr);
                    ready.store(true, Ordering::Release);
                    tx_clone.send(TrayMsg::SetState(TrayState::Off)).ok();
                });
                false
            }
            None => match device_ids.iter().find(|(id, _)| id == &event.id) {
                Some((_, name)) => {
                    for (di, n) in device_items.iter().zip(devices.iter()) {
                        di.set_checked(n == name);
                    }
                    *selected_device.lock().unwrap() = Some(name.clone());
                    audio_capture.switch_device(Some(name.as_str()));
                    false
                }
                None => false,
            },
        },
    }
}
