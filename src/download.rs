//! Downloads the model from Hugging Face into a cache directory.
//!
//! Cache location: `$XDG_CACHE_HOME/fluid-speech-rs/fluid-speech-mini-safetensors/`, falling back to
//! `~/.cache/fluid-speech-rs/...`. Each file is fetched to a `.part` sibling and renamed into place once complete,
//! so a killed download never leaves a file that looks done.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// The Hugging Face repository the model comes from.
pub const HF_REPO: &str = "apiplant/fluid-speech-mini-safetensors";
/// Files the engine needs (model.safetensors is about 327 MB).
pub const FILES: &[&str] = &["config.json", "vocab.json", "model.safetensors", "LICENSE", "NOTICE"];

/// `$XDG_CACHE_HOME/fluid-speech-rs`, defaulting to `~/.cache/fluid-speech-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("fluid-speech-rs"))
}

/// Where [`download`] puts (or would put) the model directory.
pub fn model_dir() -> Option<PathBuf> {
    Some(cache_dir()?.join(HF_REPO.rsplit('/').next().unwrap_or(HF_REPO)))
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("file").to_string();
    let part = dest.with_file_name(format!("{name}.part"));

    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());

    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let (mut written, mut last_report) = (0u64, 0u64);
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("reading body for {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
        if written - last_report >= 8 << 20 {
            last_report = written;
            match len {
                Some(len) => eprint!("\r  {name} {:.0}% ({} / {} MiB)  ", written as f64 * 100.0 / len as f64, written >> 20, len >> 20),
                None => eprint!("\r  {name} {} MiB  ", written >> 20),
            }
            std::io::stderr().flush().ok();
        }
    }
    drop(file);

    if let Some(len) = len {
        if written != len {
            let _ = std::fs::remove_file(&part);
            bail!("short read for {url}: got {written} bytes, expected {len}");
        }
    }
    std::fs::rename(&part, dest).with_context(|| format!("renaming {} to {}", part.display(), dest.display()))?;
    if written >= 8 << 20 {
        eprintln!("\r  {name} done ({} MiB)          ", written >> 20);
    }
    Ok(())
}

/// Downloads the model into the cache directory unless it is already there, and returns the directory. Needs
/// network access on first use; failures are returned as errors and nothing partial is left looking complete.
pub fn download() -> Result<PathBuf> {
    let dir = model_dir().context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    let missing: Vec<&str> = FILES.iter().copied().filter(|f| !dir.join(f).is_file()).collect();
    if missing.is_empty() {
        return Ok(dir);
    }
    eprintln!("Downloading {HF_REPO} (about 327 MB) from https://huggingface.co/{HF_REPO} into {}", dir.display());
    for f in missing {
        fetch_to_file(&format!("https://huggingface.co/{HF_REPO}/resolve/main/{f}"), &dir.join(f))?;
    }
    Ok(dir)
}
