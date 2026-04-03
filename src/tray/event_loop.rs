use crate::audio;
use crate::settings::ModelId;
use image::ImageFormat;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, Submenu};
use tray_icon::{Icon, TrayIconBuilder};

use super::menu::handle_menu_event;

pub const ICON_LOAD: &[u8] = include_bytes!("../../assets/load.png");
pub const ICON_ON: &[u8] = include_bytes!("../../assets/on.png");
pub const ICON_OFF: &[u8] = include_bytes!("../../assets/off.png");

#[repr(u8)]
pub enum TrayState {
    Loading,
    On,
    Off,
}

pub enum TrayMsg {
    SetState(TrayState),
    UpdateAvailable(String),
    Quit,
}

pub fn run_tray(
    rx: std::sync::mpsc::Receiver<TrayMsg>,
    selected_device: Arc<Mutex<Option<String>>>,
    recording_mode: Arc<AtomicU8>,
    should_quit: Arc<AtomicBool>,
    audio_capture: Arc<audio::AudioCapture>,
    models_ready: Arc<AtomicBool>,
    asr_holder: Arc<Mutex<Option<crate::asr::Asr>>>,
    tray_tx: std::sync::mpsc::Sender<TrayMsg>,
) {
    crate::platform::tray::init_app();
    let icons: [Icon; 3] = std::array::from_fn(|i| {
        let png = [ICON_LOAD, ICON_ON, ICON_OFF][i];
        let img = image::load_from_memory_with_format(png, ImageFormat::Png)
            .expect("failed to load icon PNG");
        let rgba = img.into_rgba8();
        let (w, h) = rgba.dimensions();
        Icon::from_rgba(rgba.into_raw(), w, h).expect("failed to create icon")
    });
    let devices = audio::list_input_devices();
    let current_device = selected_device.lock().unwrap().clone();
    let current_settings = crate::settings::load();
    let device_submenu = Submenu::new("Device", true);
    let device_items: Vec<CheckMenuItem> = devices
        .iter()
        .map(|name| {
            CheckMenuItem::new(
                name.as_str(),
                true,
                current_device.as_deref() == Some(name.as_str()),
                None,
            )
        })
        .collect();
    device_items.iter().for_each(|i| {
        device_submenu.append(i).ok();
    });
    let mode_submenu = Submenu::new("Mode", true);
    let use_ttt = matches!(current_settings.mode, crate::settings::Mode::ToggleToTalk);
    let ptt_item = CheckMenuItem::new("Push-to-Talk", true, !use_ttt, None);
    let ttt_item = CheckMenuItem::new("Toggle-to-Talk (double press)", true, use_ttt, None);
    let _ = mode_submenu
        .append(&ptt_item)
        .and_then(|_| mode_submenu.append(&ttt_item));
    let model_submenu = Submenu::new("Model", true);
    let model_items: Vec<CheckMenuItem> = ModelId::ALL
        .iter()
        .map(|&id| CheckMenuItem::new(id.display_name(), true, current_settings.model == id, None))
        .collect();
    model_items.iter().for_each(|i| {
        model_submenu.append(i).ok();
    });
    let version_item = MenuItem::new(&format!("vox v{}", crate::VERSION), false, None);
    let quit_item = MenuItem::new("Quit", true, None);
    let menu = Menu::new();
    let _ = menu
        .append(&version_item)
        .and_then(|_| menu.append(&device_submenu))
        .and_then(|_| menu.append(&mode_submenu))
        .and_then(|_| menu.append(&model_submenu))
        .and_then(|_| menu.append(&quit_item));
    let tray = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_icon(icons[TrayState::Loading as usize].clone())
        .with_tooltip("vox")
        .build()
        .expect("failed to create tray icon");
    let ptt_id = ptt_item.id().clone();
    let ttt_id = ttt_item.id().clone();
    let quit_id = quit_item.id().clone();
    let device_ids: Vec<_> = device_items
        .iter()
        .zip(devices.iter())
        .map(|(item, name)| (item.id().clone(), name.clone()))
        .collect();
    let model_ids: Vec<_> = model_items
        .iter()
        .zip(ModelId::ALL.iter())
        .map(|(item, &id)| (item.id().clone(), id))
        .collect();
    let mut load_cancel = Arc::new(AtomicBool::new(false));
    let handle_msg = |msg: TrayMsg| match msg {
        TrayMsg::SetState(state) => {
            tray.set_icon(Some(icons[state as usize].clone())).ok();
        }
        TrayMsg::UpdateAvailable(version) => {
            version_item.set_text(&format!("vox v{} → v{version} ↑", crate::VERSION));
        }
        TrayMsg::Quit => {
            should_quit.store(true, Ordering::Release);
        }
    };
    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(msg) => handle_msg(msg),
            Err(_) => {}
        }
        while let Ok(msg) = rx.try_recv() {
            handle_msg(msg);
        }
        match should_quit.load(Ordering::Acquire) {
            true => return,
            false => {}
        }
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            match handle_menu_event(
                event,
                &quit_id,
                &ptt_id,
                &ttt_id,
                &ptt_item,
                &ttt_item,
                &recording_mode,
                &should_quit,
                &tray,
                &icons,
                &model_ids,
                &model_items,
                asr_holder.clone(),
                models_ready.clone(),
                &mut load_cancel,
                tray_tx.clone(),
                &device_ids,
                &device_items,
                &devices,
                &selected_device,
                &audio_capture,
            ) {
                true => return,
                false => {}
            }
        }
        unsafe { crate::platform::tray::run_loop_step() };
    }
}
