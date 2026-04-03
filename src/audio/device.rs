use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Sample, SizedSample};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

pub(in crate::audio) const TARGET_SAMPLE_RATE: u32 = 16000;

pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut names: Vec<String> = host
        .input_devices()
        .map(|devs| {
            devs.filter_map(|d| d.description().ok().map(|desc| desc.name().to_string()))
                .collect()
        })
        .unwrap_or_default();
    names.sort_by_key(|name| input_device_sort_key(name));
    names
}

fn input_device_sort_key(name: &str) -> u8 {
    let n = name.to_lowercase();
    if n.contains("headset") || n.contains("headphone") || n.contains("airpod") {
        return 0;
    }
    if n.contains("bluetooth") {
        return 1;
    }
    if n.contains("built-in") || n.contains("macbook") || n.contains("internal") {
        return 3;
    }
    2
}

pub(in crate::audio) fn to_mono_f32_into<T: cpal::Sample>(
    samples: &[T],
    channels: u16,
    out: &mut Vec<f32>,
) where
    f32: cpal::FromSample<T>,
{
    if channels == 1 {
        out.extend(samples.iter().map(|s| f32::from_sample(*s)));
        return;
    }
    let ch = channels as usize;
    out.extend(samples.chunks_exact(ch).map(|chunk| {
        let sum: f32 = chunk.iter().map(|s| f32::from_sample(*s)).sum();
        sum / ch as f32
    }));
}

pub(in crate::audio) fn build_stream<T: Sample + SizedSample + Send + 'static>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: u16,
    producer: Arc<Mutex<rtrb::Producer<f32>>>,
    is_active: Arc<AtomicBool>,
    data_signal: Arc<(Mutex<bool>, Condvar)>,
) -> cpal::Stream
where
    f32: cpal::FromSample<T>,
{
    let mut mono_buf: Vec<f32> = Vec::with_capacity(4096);
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                if !is_active.load(Ordering::Acquire) {
                    return;
                }
                mono_buf.clear();
                to_mono_f32_into(data, channels, &mut mono_buf);
                if let Ok(mut p) = producer.try_lock() {
                    p.push_partial_slice(&mono_buf);
                }
                let (lock, cvar) = &*data_signal;
                if let Ok(mut ready) = lock.try_lock() {
                    *ready = true;
                    cvar.notify_one();
                }
            },
            |_err: cpal::StreamError| {},
            None,
        )
        .expect("failed to build input stream")
}
