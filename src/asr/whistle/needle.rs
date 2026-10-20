#[cfg(target_arch = "aarch64")]
use std::ffi::{c_char, c_int, c_ulonglong, CStr};
#[cfg(target_arch = "aarch64")]
use std::ptr::null;

#[cfg(target_arch = "aarch64")]
extern "C" {
    fn needle_load(model_bytes: *const u8, model_length: c_ulonglong) -> c_int;
    fn needle_last_error() -> *const c_char;
    fn needle_transcribe(
        pcm: *const f32,
        samples: c_int,
        language: *const c_char,
        keywords: *const c_char,
        word_timestamps: c_int,
        output: *mut u8,
        output_capacity: c_int,
    ) -> c_int;
}

#[cfg(target_arch = "aarch64")]
fn check(status_code: c_int) -> Result<(), String> {
    if status_code >= 0 {
        return Ok(());
    }
    let error_pointer = unsafe { needle_last_error() };
    if error_pointer.is_null() {
        return Err("unknown error".into());
    }
    Err(unsafe { CStr::from_ptr(error_pointer) }
        .to_string_lossy()
        .into_owned())
}

#[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
pub fn load(model_bytes: &[u8]) -> Result<(), String> {
    #[cfg(not(target_arch = "aarch64"))]
    return Err("Not supported on the current platform".into());
    #[cfg(target_arch = "aarch64")]
    check(unsafe { needle_load(model_bytes.as_ptr(), model_bytes.len() as c_ulonglong) })
}

#[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
pub fn transcribe(audio: &[f32], output: &mut [u8]) -> Result<(), String> {
    #[cfg(not(target_arch = "aarch64"))]
    return Err("Not supported on the current platform".into());
    #[cfg(target_arch = "aarch64")]
    check(unsafe {
        needle_transcribe(
            audio.as_ptr(),
            audio.len() as c_int,
            null(),
            null(),
            0,
            output.as_mut_ptr(),
            output.len() as c_int,
        )
    })
}
