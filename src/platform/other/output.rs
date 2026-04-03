use std::thread;
use std::time::Duration;

pub fn simulate_paste() {
    use rdev::{simulate, EventType, Key};

    let events = [
        EventType::KeyPress(Key::ControlLeft),
        EventType::KeyPress(Key::KeyV),
        EventType::KeyRelease(Key::KeyV),
        EventType::KeyRelease(Key::ControlLeft),
    ];

    for event in &events {
        let _ = simulate(event);
        thread::sleep(Duration::from_millis(20));
    }
}
