use arboard::Clipboard;
use std::thread;
use std::time::Duration;

pub fn paste_at_cursor(text: &str) {
    if text.is_empty() {
        return;
    }

    let mut clipboard = match Clipboard::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    let previous = clipboard.get_text().ok();

    if clipboard.set_text(text).is_err() {
        return;
    }

    thread::sleep(Duration::from_millis(20));
    crate::platform::simulate_paste();
    thread::sleep(Duration::from_millis(100));

    if let Some(prev) = previous {
        let _ = clipboard.set_text(prev);
    }
}
