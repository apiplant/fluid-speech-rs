//! Linux backend: hotkeys from /dev/input (evdev), typing through /dev/uinput, `wl-copy`, `notify-send`.
//!
//! Works under any compositor (Wayland or X11). Both device interfaces need the user in the `input` group
//! (or equivalent udev rules).

use anyhow::{Context, Result, bail};
use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, EventSummary, KeyCode};
use std::collections::{BTreeSet, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::key::{Key, canonical, is_modifier, us_key};

const VIRTUAL_KBD: &str = "fluid-dictate virtual keyboard";

/// The shortcut that pastes.
pub const DEFAULT_PASTE_KEYS: &str = "ctrl+v";

/// What to tell the user when a hotkey cannot be read.
pub const PERMISSION_HINT: &str = "cannot read any input device in /dev/input. Add yourself to the `input` group \
    (sudo usermod -aG input $USER, then log in again) or run with --no-hotkey";

pub fn valid_name(name: &str) -> bool {
    KeyCode::from_str(name).is_ok()
}

fn to_evdev(k: Key) -> Result<KeyCode> {
    KeyCode::from_str(k.name()).map_err(|_| anyhow::anyhow!("no such key {}", k.name()))
}

fn from_evdev(code: KeyCode) -> Option<Key> {
    // evdev names its keys exactly as the config does; codes it has no name for are not hotkey material
    Key::from_name(&format!("{code:?}")).map(canonical)
}

/// Key down/up events (canonicalised, repeats dropped) from every keyboard-like device, including ones plugged
/// in later.
pub fn spawn_key_listener() -> Result<Receiver<(Key, bool)>> {
    let (tx, rx) = channel();
    let open: Arc<Mutex<HashSet<PathBuf>>> = Arc::default();
    let found = scan_devices(&tx, &open);
    if found == 0 {
        bail!("{PERMISSION_HINT}");
    }
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(2));
            scan_devices(&tx, &open);
        }
    });
    Ok(rx)
}

fn scan_devices(tx: &Sender<(Key, bool)>, open: &Arc<Mutex<HashSet<PathBuf>>>) -> usize {
    for (path, mut dev) in evdev::enumerate() {
        if open.lock().unwrap().contains(&path) || dev.name() == Some(VIRTUAL_KBD) {
            continue;
        }
        // anything with keys or buttons: keyboards, macro pads, mouse side buttons
        let has_keys = dev.supported_keys().is_some_and(|k| k.iter().next().is_some());
        if !has_keys {
            continue;
        }
        log::debug!("listening on {} ({})", path.display(), dev.name().unwrap_or("?"));
        open.lock().unwrap().insert(path.clone());
        let (tx, open) = (tx.clone(), open.clone());
        std::thread::spawn(move || {
            'outer: loop {
                let events = match dev.fetch_events() {
                    Ok(e) => e,
                    Err(_) => break, // unplugged
                };
                for ev in events {
                    if let EventSummary::Key(_, code, v) = ev.destructure() {
                        if v == 2 {
                            continue; // autorepeat
                        }
                        let Some(k) = from_evdev(code) else { continue };
                        if tx.send((k, v == 1)).is_err() {
                            break 'outer;
                        }
                    }
                }
            }
            open.lock().unwrap().remove(&path);
        });
    }
    open.lock().unwrap().len()
}

/// Types text through a virtual keyboard (US layout).
pub struct Typer {
    dev: VirtualDevice,
}

impl Typer {
    pub fn new() -> Result<Self> {
        let mut keys = AttributeSet::<KeyCode>::new();
        for c in (32u8..127).map(char::from).chain(['\n', '\t']) {
            if let Some((k, _)) = us_key(c) {
                keys.insert(to_evdev(k)?);
            }
        }
        for k in [KeyCode::KEY_LEFTSHIFT, KeyCode::KEY_LEFTCTRL, KeyCode::KEY_LEFTALT, KeyCode::KEY_LEFTMETA, KeyCode::KEY_V, KeyCode::KEY_INSERT] {
            keys.insert(k);
        }
        let dev = VirtualDevice::builder()
            .context("cannot open /dev/uinput (needs the `input` group or a udev rule); try --output paste/stdout")?
            .name(VIRTUAL_KBD)
            .with_keys(&keys)?
            .build()?;
        Ok(Self { dev })
    }

    /// Whether `c` can be typed directly; anything else goes through the clipboard.
    pub fn can_type(&self, c: char) -> bool {
        us_key(c).is_some()
    }

    fn tap(&mut self, k: KeyCode, mods: &[KeyCode]) -> Result<()> {
        let ev = |k: KeyCode, v: i32| *evdev::KeyEvent::new(k, v);
        for &m in mods {
            self.dev.emit(&[ev(m, 1)])?;
        }
        self.dev.emit(&[ev(k, 1)])?;
        self.dev.emit(&[ev(k, 0)])?;
        for &m in mods.iter().rev() {
            self.dev.emit(&[ev(m, 0)])?;
        }
        // compositors drop keys that arrive faster than they process them
        std::thread::sleep(Duration::from_millis(2));
        Ok(())
    }

    pub fn type_text(&mut self, text: &str) -> Result<()> {
        for c in text.chars() {
            let (k, shift) = us_key(c).with_context(|| format!("cannot type {c:?}"))?;
            let k = to_evdev(k)?;
            if shift {
                self.tap(k, &[KeyCode::KEY_LEFTSHIFT])?;
            } else {
                self.tap(k, &[])?;
            }
        }
        Ok(())
    }

    /// Presses a shortcut such as Ctrl+V.
    pub fn press_combo(&mut self, keys: &BTreeSet<Key>) -> Result<()> {
        let (mods, main): (Vec<Key>, Vec<Key>) = keys.iter().partition(|k| is_modifier(**k));
        let main = *main.first().context("a paste shortcut needs a non-modifier key")?;
        let mods: Vec<KeyCode> = mods.into_iter().map(to_evdev).collect::<Result<_>>()?;
        self.tap(to_evdev(main)?, &mods)
    }
}

pub fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut child = std::process::Command::new("wl-copy")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("wl-copy not found (install wl-clipboard)")?;
    child.stdin.take().unwrap().write_all(text.as_bytes())?;
    child.wait()?;
    Ok(())
}

/// Desktop notifications through `notify-send`, replacing the previous one.
pub struct Notifier {
    enabled: bool,
    id: Option<String>,
}

impl Notifier {
    pub fn new(enabled: bool) -> Self {
        Self { enabled, id: None }
    }

    pub fn show(&mut self, summary: &str, body: &str, ms: u32) {
        if !self.enabled {
            return;
        }
        let mut cmd = std::process::Command::new("notify-send");
        cmd.args(["-a", "Fluid dictation", "-i", "audio-input-microphone", "-t", &ms.to_string(), "-p"]);
        if let Some(id) = &self.id {
            cmd.args(["-r", id]);
        }
        cmd.arg(summary);
        if !body.is_empty() {
            cmd.arg(body);
        }
        if let Ok(out) = cmd.output() {
            let id = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !id.is_empty() {
                self.id = Some(id);
            }
        }
    }
}
