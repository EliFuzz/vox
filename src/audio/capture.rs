use super::device::{build_stream, TARGET_SAMPLE_RATE};
use audioadapter_buffers::direct::InterleavedSlice;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

const RING_CAPACITY: usize = 160_000;

fn resampler_and_chunk_size(native_sample_rate: u32) -> (Option<Async<f32>>, usize) {
    if native_sample_rate == TARGET_SAMPLE_RATE {
        return (None, 0);
    }
    let params = SincInterpolationParameters {
        sinc_len: 32,
        f_cutoff: 0.91,
        interpolation: SincInterpolationType::Nearest,
        oversampling_factor: 64,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk_size = (native_sample_rate as f64 * 0.03) as usize;
    let r = Async::<f32>::new_sinc(
        TARGET_SAMPLE_RATE as f64 / native_sample_rate as f64,
        2.0,
        &params,
        chunk_size,
        1,
        FixedAsync::Input,
    )
    .expect("failed to create resampler");
    (Some(r), chunk_size)
}

pub struct AudioCapture {
    device_name: Mutex<Option<String>>,
    stream: Mutex<Option<cpal::Stream>>,
    producer: Mutex<Option<rtrb::Producer<f32>>>,
    consumer: Mutex<Option<rtrb::Consumer<f32>>>,
    out_buffer: Mutex<Vec<f32>>,
    is_active: Arc<AtomicBool>,
    resampler: Mutex<Option<Async<f32>>>,
    resample_buf: Mutex<Vec<f32>>,
    resample_chunk_size: Mutex<usize>,
    data_ready: Arc<(Mutex<bool>, Condvar)>,
}
unsafe impl Send for AudioCapture {}
unsafe impl Sync for AudioCapture {}

impl AudioCapture {
    pub fn new_with_device(device_name: Option<&str>) -> Self {
        let (producer, consumer) = rtrb::RingBuffer::new(RING_CAPACITY);
        Self {
            device_name: Mutex::new(device_name.map(|s| s.to_string())),
            stream: Mutex::new(None),
            producer: Mutex::new(Some(producer)),
            consumer: Mutex::new(Some(consumer)),
            out_buffer: Mutex::new(Vec::with_capacity(TARGET_SAMPLE_RATE as usize * 10)),
            is_active: Arc::new(AtomicBool::new(false)),
            resampler: Mutex::new(None),
            resample_buf: Mutex::new(Vec::with_capacity(32_000)),
            resample_chunk_size: Mutex::new(0),
            data_ready: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }
    pub fn data_ready_signal(&self) -> Arc<(Mutex<bool>, Condvar)> {
        self.data_ready.clone()
    }
    fn init_stream(&self) {
        let device_name = self.device_name.lock().unwrap().clone();
        let host = cpal::default_host();
        let device = match device_name.as_deref() {
            Some(name) => host
                .input_devices()
                .ok()
                .and_then(|mut devs| {
                    devs.find(|d| {
                        d.description()
                            .ok()
                            .map(|desc| desc.name().to_string())
                            .as_deref()
                            == Some(name)
                    })
                })
                .unwrap_or_else(|| host.default_input_device().expect("no input device")),
            None => host.default_input_device().expect("no input device"),
        };
        let config = device
            .default_input_config()
            .expect("failed to get input config");
        let native_sample_rate = config.sample_rate();
        let native_channels = config.channels();
        let (new_resampler, chunk_size) = resampler_and_chunk_size(native_sample_rate);
        *self.resampler.lock().unwrap() = new_resampler;
        *self.resample_chunk_size.lock().unwrap() = chunk_size;
        let (producer, consumer) = rtrb::RingBuffer::new(RING_CAPACITY);
        *self.producer.lock().unwrap() = Some(producer);
        *self.consumer.lock().unwrap() = Some(consumer);
        let producer_for_stream = {
            let mut guard = self.producer.lock().unwrap();
            guard.take().expect("producer missing")
        };
        let producer_cell = Arc::new(Mutex::new(producer_for_stream));
        let active_clone = self.is_active.clone();
        let data_signal = self.data_ready.clone();
        let built_stream = match config.sample_format() {
            SampleFormat::F32 => build_stream::<f32>(
                &device,
                &config.into(),
                native_channels,
                producer_cell,
                active_clone,
                data_signal,
            ),
            SampleFormat::I16 => build_stream::<i16>(
                &device,
                &config.into(),
                native_channels,
                producer_cell,
                active_clone,
                data_signal,
            ),
            SampleFormat::I32 => build_stream::<i32>(
                &device,
                &config.into(),
                native_channels,
                producer_cell,
                active_clone,
                data_signal,
            ),
            f => panic!("unsupported sample format: {f}"),
        };
        *self.stream.lock().unwrap() = Some(built_stream);
    }
    pub fn switch_device(&self, device_name: Option<&str>) {
        self.is_active.store(false, Ordering::Release);
        if let Some(s) = self.stream.lock().unwrap().take() {
            drop(s);
        }
        let (producer, consumer) = rtrb::RingBuffer::new(RING_CAPACITY);
        *self.producer.lock().unwrap() = Some(producer);
        *self.consumer.lock().unwrap() = Some(consumer);
        self.out_buffer.lock().unwrap().clear();
        self.resample_buf.lock().unwrap().clear();
        *self.resampler.lock().unwrap() = None;
        *self.device_name.lock().unwrap() = device_name.map(|s| s.to_string());
    }
    pub fn start(&self) {
        if self.stream.lock().unwrap().is_none() {
            self.init_stream();
        }
        let guard = self.stream.lock().unwrap();
        if let Some(stream) = guard.as_ref() {
            self.out_buffer.lock().unwrap().clear();
            self.resample_buf.lock().unwrap().clear();
            self.is_active.store(true, Ordering::Release);
            stream.play().expect("failed to start audio stream");
        }
    }
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
        let guard = self.stream.lock().unwrap();
        if let Some(stream) = guard.as_ref() {
            stream.pause().ok();
        }
    }
    pub fn drain_chunk(&self, max_samples: usize) -> Vec<f32> {
        {
            let mut consumer_guard = self.consumer.lock().unwrap();
            if let Some(ref mut consumer) = *consumer_guard {
                let available = consumer.slots();
                if available > 0 {
                    let mut rbuf = self.resample_buf.lock().unwrap();
                    if let Ok(chunk) = consumer.read_chunk(available) {
                        let (first, second) = chunk.as_slices();
                        rbuf.extend_from_slice(first);
                        rbuf.extend_from_slice(second);
                        chunk.commit_all();
                    }
                }
            }
        }
        let chunk_size = *self.resample_chunk_size.lock().unwrap();
        let mut resampler = self.resampler.lock().unwrap();
        let no_resampler = resampler.is_none();
        if let Some(ref mut r) = *resampler {
            let mut rbuf = self.resample_buf.lock().unwrap();
            let mut out = self.out_buffer.lock().unwrap();
            while rbuf.len() >= chunk_size {
                let chunk: Vec<f32> = rbuf.drain(..chunk_size).collect();
                let adapter =
                    InterleavedSlice::new(&chunk, 1, chunk_size).expect("InterleavedSlice failed");
                if let Ok(result) = r.process(&adapter, 0, None) {
                    let samples = result.take_data();
                    out.extend_from_slice(&samples);
                }
            }
        }
        if no_resampler {
            let mut rbuf = self.resample_buf.lock().unwrap();
            if !rbuf.is_empty() {
                self.out_buffer.lock().unwrap().extend_from_slice(&rbuf);
                rbuf.clear();
            }
        }
        let mut out = self.out_buffer.lock().unwrap();
        let n = max_samples.min(out.len());
        out.drain(..n).collect()
    }
}
