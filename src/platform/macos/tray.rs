pub fn init_app() {
    unsafe {
        #[link(name = "AppKit", kind = "framework")]
        extern "C" {
            static NSApp: *mut objc2::runtime::AnyObject;
        }
        let app_cls = objc2::runtime::AnyClass::get(c"NSApplication").unwrap();
        let _: *mut objc2::runtime::AnyObject = objc2::msg_send![app_cls, sharedApplication];
        let _: bool = objc2::msg_send![NSApp, setActivationPolicy: 1i64];
    }
}

pub unsafe fn run_loop_step() {
    #[link(name = "AppKit", kind = "framework")]
    extern "C" {
        static NSApp: *mut objc2::runtime::AnyObject;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        static kCFRunLoopDefaultMode: *const std::ffi::c_void;
        fn CFRunLoopRunInMode(
            mode: *const std::ffi::c_void,
            seconds: f64,
            returnAfterSourceHandled: u8,
        ) -> i32;
    }
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.0, false as u8);
    let mask: u64 = u64::MAX;
    let date_cls = objc2::runtime::AnyClass::get(c"NSDate").unwrap();
    let dist_past: *mut objc2::runtime::AnyObject = objc2::msg_send![date_cls, distantPast];
    let mode_str = objc2_foundation::NSString::from_str("kCFRunLoopDefaultMode");
    loop {
        let event: *mut objc2::runtime::AnyObject = objc2::msg_send![
            NSApp,
            nextEventMatchingMask: mask,
            untilDate: dist_past,
            inMode: &*mode_str,
            dequeue: true
        ];
        if event.is_null() {
            break;
        }
        let () = objc2::msg_send![NSApp, sendEvent: event];
    }
}
