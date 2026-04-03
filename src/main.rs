mod asr;
mod audio;
mod model_download;
mod normalization;
mod output;
mod platform;
mod processing;
mod recording;
mod settings;
mod tray;
mod vad;

#[cfg(test)]
mod tests;

use self_update::backends::github::ReleaseList;
use self_update::version::bump_is_greater;
use settings::ModelId;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

type AsrHolder = Arc<Mutex<Option<asr::Asr>>>;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const MODE_PTT: u8 = 1;
pub const MODE_TTT: u8 = 2;

static QUIT_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static CTRLC_TRAY_TX: OnceLock<std::sync::mpsc::Sender<tray::TrayMsg>> = OnceLock::new();

fn register_ctrlc(quit: Arc<AtomicBool>, tx: std::sync::mpsc::Sender<tray::TrayMsg>) {
    QUIT_FLAG.get_or_init(|| quit);
    CTRLC_TRAY_TX.get_or_init(|| tx);

    extern "C" fn handler(_: i32) {
        if let Some(flag) = QUIT_FLAG.get() {
            flag.store(true, Ordering::Release);
        }
        if let Some(tx) = CTRLC_TRAY_TX.get() {
            tx.send(tray::TrayMsg::Quit).ok();
        }
        std::process::exit(0);
    }

    extern "C" {
        fn signal(sig: i32, handler: usize) -> usize;
    }

    unsafe {
        signal(2, handler as usize);
    }
}

const UPDATE_CHECK_INTERVAL: Duration = Duration::from_secs(86400);

fn start_updater(tx: std::sync::mpsc::Sender<tray::TrayMsg>, should_quit: Arc<AtomicBool>) {
    std::thread::spawn(move || loop {
        if should_quit.load(Ordering::Acquire) {
            return;
        }
        if let Some(version) = fetch_latest_version() {
            if bump_is_greater(VERSION, &version).unwrap_or(false) {
                tx.send(tray::TrayMsg::UpdateAvailable(version)).ok();
            }
        }
        let deadline = std::time::Instant::now() + UPDATE_CHECK_INTERVAL;
        while std::time::Instant::now() < deadline {
            if should_quit.load(Ordering::Acquire) {
                return;
            }
            std::thread::sleep(Duration::from_secs(60));
        }
    });
}

fn fetch_latest_version() -> Option<String> {
    ReleaseList::configure()
        .repo_owner(model_download::REPO_OWNER)
        .repo_name(model_download::REPO_NAME)
        .build()
        .ok()?
        .fetch()
        .ok()?
        .into_iter()
        .next()
        .map(|r| r.version)
}

fn load_asr_model(id: ModelId) -> asr::Asr {
    let dir = settings::model_path(id);
    assert!(
        dir.exists(),
        "model not found: {}. Place models at {}",
        id.as_str(),
        dir.display()
    );
    asr::Asr::new(id, &dir)
}

fn main() {
    if !platform::check_accessibility() {
        eprintln!("Accessibility permission required: grant access in System Settings > Privacy & Security > Accessibility, then restart vox");
        std::process::exit(1);
    }

    let config = settings::load();

    let is_recording = Arc::new(AtomicBool::new(false));
    let initial_mode = match config.mode {
        settings::Mode::ToggleToTalk => MODE_TTT,
        settings::Mode::PushToTalk => MODE_PTT,
    };
    let recording_mode = Arc::new(AtomicU8::new(initial_mode));
    let should_quit = Arc::new(AtomicBool::new(false));
    let models_ready = Arc::new(AtomicBool::new(false));

    let vad_holder: Arc<OnceLock<Mutex<vad::Vad>>> = Arc::new(OnceLock::new());
    let asr_holder: AsrHolder = Arc::new(Mutex::new(None));

    let default_device = config
        .input_device
        .clone()
        .or_else(|| audio::list_input_devices().into_iter().next());
    let selected_device = Arc::new(Mutex::new(default_device.clone()));
    let audio_capture = Arc::new(audio::AudioCapture::new_with_device(
        default_device.as_deref(),
    ));

    let (tray_tx, tray_rx) = std::sync::mpsc::channel::<tray::TrayMsg>();

    let vad_h = vad_holder.clone();
    let asr_h = asr_holder.clone();
    let ready = models_ready.clone();
    let tx = tray_tx.clone();
    let model_id = config.model;
    std::thread::spawn(move || {
        if let Err(e) = model_download::ensure_vad() {
            eprintln!("vad download failed: {e}");
        }
        let vad_path = settings::vad_model_path();
        if vad_path.exists() {
            vad_h.get_or_init(|| Mutex::new(vad::Vad::new(&vad_path, 0.5)));
        }

        if settings::model_ready(model_id) {
            let asr = load_asr_model(model_id);
            *asr_h.lock().unwrap() = Some(asr);
            ready.store(true, Ordering::Release);
        }
        tx.send(tray::TrayMsg::SetState(tray::TrayState::Off)).ok();
    });

    let (a, v, r, is_rec, rdy, quit, tx) = (
        audio_capture.clone(),
        vad_holder.clone(),
        asr_holder.clone(),
        is_recording.clone(),
        models_ready.clone(),
        should_quit.clone(),
        tray_tx.clone(),
    );
    std::thread::spawn(move || processing::run(a, v, r, is_rec, rdy, quit, tx));

    let (is_rec, mode, aud, rdy, quit, tx) = (
        is_recording.clone(),
        recording_mode.clone(),
        audio_capture.clone(),
        models_ready.clone(),
        should_quit.clone(),
        tray_tx.clone(),
    );
    std::thread::spawn(move || platform::keyboard::listen(is_rec, mode, aud, rdy, quit, tx));

    register_ctrlc(should_quit.clone(), tray_tx.clone());
    start_updater(tray_tx.clone(), should_quit.clone());
    tray::run_tray(
        tray_rx,
        selected_device,
        recording_mode,
        should_quit,
        audio_capture,
        models_ready,
        asr_holder,
        tray_tx,
    );
}
