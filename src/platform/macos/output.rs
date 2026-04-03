use std::thread;
use std::time::Duration;

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventSourceCreate(state_id: i32) -> *mut std::ffi::c_void;
    fn CGEventCreateKeyboardEvent(
        source: *mut std::ffi::c_void,
        virtual_key: u16,
        key_down: bool,
    ) -> *mut std::ffi::c_void;
    fn CGEventSetFlags(event: *mut std::ffi::c_void, flags: u64);
    fn CGEventPost(tap: u32, event: *mut std::ffi::c_void);
    fn CFRelease(cf: *mut std::ffi::c_void);
}

const CG_SESSION_EVENT_TAP: u32 = 1;
const CG_EVENT_SOURCE_STATE_PRIVATE: i32 = -1;
const CG_EVENT_FLAG_MASK_COMMAND: u64 = 0x0010_0000;
const VK_ANSI_V: u16 = 9;

unsafe fn post_cmd_v(source: *mut std::ffi::c_void, key_down: bool) {
    let ev = CGEventCreateKeyboardEvent(source, VK_ANSI_V, key_down);
    if !ev.is_null() {
        CGEventSetFlags(ev, CG_EVENT_FLAG_MASK_COMMAND);
        CGEventPost(CG_SESSION_EVENT_TAP, ev);
        CFRelease(ev);
    }
}

pub fn simulate_paste() {
    unsafe {
        let source = CGEventSourceCreate(CG_EVENT_SOURCE_STATE_PRIVATE);
        if source.is_null() {
            return;
        }

        post_cmd_v(source, true);
        thread::sleep(Duration::from_millis(20));
        post_cmd_v(source, false);

        CFRelease(source);
    }
}
