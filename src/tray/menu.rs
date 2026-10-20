use super::{TrayMsg, TrayState};
use crate::audio;
use crate::settings::ModelId;
use crate::settings::{load, model_path, update, Mode};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use tray_icon::menu::{CheckMenuItem, MenuEvent, MenuId};
use tray_icon::{Icon, TrayIcon};

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
            update(|settings| settings.mode = Mode::PushToTalk);
            false
        }
        id if id == ttt_id => {
            ttt_item.set_checked(true);
            ptt_item.set_checked(false);
            recording_mode.store(crate::MODE_TTT, Ordering::Release);
            update(|settings| settings.mode = Mode::ToggleToTalk);
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
                load_cancel.store(true, Ordering::Release);
                if asr_holder.lock().unwrap().is_some() && load().model == model_id {
                    tray_tx.send(TrayMsg::SetState(TrayState::Off)).ok();
                    return false;
                }
                let asr_h = asr_holder.clone();
                let ready = models_ready.clone();
                let tx_clone = tray_tx.clone();
                tray.set_icon(Some(icons[TrayState::Loading as usize].clone()))
                    .ok();
                let cancel = Arc::new(AtomicBool::new(false));
                *load_cancel = cancel.clone();
                std::thread::spawn(move || {
                    if let Err(e) = crate::model_download::ensure_model(model_id) {
                        eprintln!("model download failed for {}: {e}", model_id.as_str());
                        if !cancel.load(Ordering::Acquire) {
                            tx_clone.send(TrayMsg::SetState(TrayState::Off)).ok();
                        }
                        return;
                    }
                    if cancel.load(Ordering::Acquire) {
                        return;
                    }
                    let dir = model_path(model_id);
                    let asr = crate::asr::Asr::new(model_id, &dir);
                    let mut holder = asr_h.lock().unwrap();
                    if cancel.load(Ordering::Acquire) {
                        return;
                    }
                    *holder = Some(asr);
                    update(|settings| settings.model = model_id);
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
                    update(|settings| settings.input_device = Some(name.clone()));
                    false
                }
                None => false,
            },
        },
    }
}
