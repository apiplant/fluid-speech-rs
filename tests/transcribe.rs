//! End-to-end checks. They need the exported model in `model/` (see README) and are skipped otherwise.

use candle_core::Device;
use fluid_speech::{audio, Engine};
use std::path::Path;

const JFK: &str = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

fn model() -> Option<&'static Path> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("model");
    if p.join("model.safetensors").exists() {
        Some(Box::leak(p.into_boxed_path()))
    } else {
        eprintln!("model/ not found, skipping");
        None
    }
}

fn sample(name: &str) -> Vec<f32> {
    audio::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("samples").join(name)).unwrap()
}

fn check(device: Device) {
    let Some(dir) = model() else { return };
    let engine = Engine::load(dir, &device).unwrap();
    let jfk = sample("jfk.wav");
    let r = engine.transcribe(&jfk).unwrap();
    assert_eq!(r.text, JFK);
    // word timestamps are monotonic and inside the clip
    assert_eq!(r.words.len(), 22);
    assert!(r.words.windows(2).all(|w| w[0].start <= w[1].start));
    assert!(r.words.last().unwrap().end <= 11.1);
    assert_eq!(fluid_speech::text::join(&r.words), JFK);
    // a clean recording decodes with high confidence
    assert!(r.words.iter().all(|w| w.confidence > 0.5), "{:?}", r.words);
    assert!(fluid_speech::text::confidence(&r.words) > 0.9);
    // batching a long and a short clip together gives the same results as one at a time
    let short = &jfk[..16000 * 3];
    let alone = engine.transcribe(short).unwrap().text;
    let batch = engine.transcribe_many(&[&jfk, short]).unwrap();
    assert_eq!(batch[0].text, JFK);
    assert_eq!(batch[1].text, alone);
    // silence and very short input
    assert_eq!(engine.transcribe(&vec![0f32; 16000]).unwrap().text, "");
    assert_eq!(engine.transcribe(&[0f32; 100]).unwrap().text, "");
}

#[test]
fn cpu() {
    check(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda() {
    check(Device::new_cuda(0).unwrap());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_long_audio_is_split_and_batched() {
    let Some(dir) = model() else { return };
    let engine = Engine::load(dir, &Device::new_cuda(0).unwrap()).unwrap();
    // 100 s of repeated speech: split into pieces at quiet points, encoded as one batch
    let jfk = sample("jfk.wav");
    let long: Vec<f32> = (0..9).flat_map(|_| jfk.iter().copied().chain(std::iter::repeat_n(0f32, 2000))).collect();
    let r = engine.transcribe(&long).unwrap();
    assert!(engine.split(&long).len() > 1);
    assert_eq!(r.text.matches("ask what you can do for your country").count(), 9, "{}", r.text);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_concurrent_engines() {
    let Some(dir) = model() else { return };
    let jfk = sample("jfk.wav");
    std::thread::scope(|s| {
        for t in 0..4 {
            let jfk = &jfk;
            s.spawn(move || {
                let engine = Engine::load(dir, &Device::new_cuda(0).unwrap()).unwrap();
                for i in 0..3 {
                    // different lengths -> new graph captures while the other threads are busy
                    let n = 16000 * (2 + (t + i) % 9);
                    assert_eq!(engine.transcribe(jfk).unwrap().text, JFK);
                    engine.transcribe(&jfk[..n]).unwrap();
                }
            });
        }
    });
}
