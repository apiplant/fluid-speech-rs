//! macOS backend: hotkeys from a listen-only Quartz event tap, text typed (or pasted) with posted key events,
//! the clipboard through `pbcopy`, notifications through `osascript`.
//!
//! macOS gates all of that behind privacy permissions, granted to the application that launches
//! `fluid-dictate` (Terminal, iTerm, ...) in System Settings > Privacy & Security:
//! **Input Monitoring** (to see the hotkey) and **Accessibility** (to type into other windows).

use anyhow::{Context, Result, bail};
use core_foundation::runloop::CFRunLoop;
use core_graphics::event::{
    CGEvent, CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventType,
    CallbackResult, EventField,
};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use std::collections::{BTreeSet, HashSet};
use std::io::Write;
use std::sync::mpsc::{Receiver, channel};
use std::sync::Mutex;
use std::time::Duration;

use super::key::{Key, canonical, is_modifier};

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXIsProcessTrusted() -> bool;
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGPreflightListenEventAccess() -> bool;
    fn CGRequestListenEventAccess() -> bool;
}

/// The shortcut that pastes.
pub const DEFAULT_PASTE_KEYS: &str = "cmd+v";

/// What to tell the user when the hotkey cannot be read.
pub const PERMISSION_HINT: &str = "cannot watch the keyboard. Allow your terminal (the app running fluid-dictate) \
    under System Settings > Privacy & Security > Input Monitoring, then start it again, or run with --no-hotkey \
    and bind `fluid-dictate toggle` in a shortcut tool";

/// macOS virtual key codes (ANSI layout) and the evdev-style names the config uses.
const KEYCODES: &[(u16, &str)] = &[
    (0, "KEY_A"), (11, "KEY_B"), (8, "KEY_C"), (2, "KEY_D"), (14, "KEY_E"), (3, "KEY_F"), (5, "KEY_G"), (4, "KEY_H"),
    (34, "KEY_I"), (38, "KEY_J"), (40, "KEY_K"), (37, "KEY_L"), (46, "KEY_M"), (45, "KEY_N"), (31, "KEY_O"),
    (35, "KEY_P"), (12, "KEY_Q"), (15, "KEY_R"), (1, "KEY_S"), (17, "KEY_T"), (32, "KEY_U"), (9, "KEY_V"),
    (13, "KEY_W"), (7, "KEY_X"), (16, "KEY_Y"), (6, "KEY_Z"),
    (29, "KEY_0"), (18, "KEY_1"), (19, "KEY_2"), (20, "KEY_3"), (21, "KEY_4"), (23, "KEY_5"), (22, "KEY_6"),
    (26, "KEY_7"), (28, "KEY_8"), (25, "KEY_9"),
    (27, "KEY_MINUS"), (24, "KEY_EQUAL"), (33, "KEY_LEFTBRACE"), (30, "KEY_RIGHTBRACE"), (42, "KEY_BACKSLASH"),
    (41, "KEY_SEMICOLON"), (39, "KEY_APOSTROPHE"), (50, "KEY_GRAVE"), (43, "KEY_COMMA"), (47, "KEY_DOT"),
    (44, "KEY_SLASH"),
    (49, "KEY_SPACE"), (36, "KEY_ENTER"), (48, "KEY_TAB"), (51, "KEY_BACKSPACE"), (53, "KEY_ESC"),
    (117, "KEY_DELETE"), (114, "KEY_INSERT"), (115, "KEY_HOME"), (119, "KEY_END"), (116, "KEY_PAGEUP"),
    (121, "KEY_PAGEDOWN"), (123, "KEY_LEFT"), (124, "KEY_RIGHT"), (125, "KEY_DOWN"), (126, "KEY_UP"),
    (122, "KEY_F1"), (120, "KEY_F2"), (99, "KEY_F3"), (118, "KEY_F4"), (96, "KEY_F5"), (97, "KEY_F6"),
    (98, "KEY_F7"), (100, "KEY_F8"), (101, "KEY_F9"), (109, "KEY_F10"), (103, "KEY_F11"), (111, "KEY_F12"),
    (105, "KEY_F13"), (107, "KEY_F14"), (113, "KEY_F15"), (106, "KEY_F16"), (64, "KEY_F17"), (79, "KEY_F18"),
    (80, "KEY_F19"), (90, "KEY_F20"),
    (59, "KEY_LEFTCTRL"), (62, "KEY_RIGHTCTRL"), (56, "KEY_LEFTSHIFT"), (60, "KEY_RIGHTSHIFT"),
    (58, "KEY_LEFTALT"), (61, "KEY_RIGHTALT"), (55, "KEY_LEFTMETA"), (54, "KEY_RIGHTMETA"),
    (63, "KEY_FN"),
];

/// Modifier key codes and the flag each one sets in `FlagsChanged` events.
const MODIFIERS: &[(u16, CGEventFlags)] = &[
    (59, CGEventFlags::CGEventFlagControl),
    (62, CGEventFlags::CGEventFlagControl),
    (56, CGEventFlags::CGEventFlagShift),
    (60, CGEventFlags::CGEventFlagShift),
    (58, CGEventFlags::CGEventFlagAlternate),
    (61, CGEventFlags::CGEventFlagAlternate),
    (55, CGEventFlags::CGEventFlagCommand),
    (54, CGEventFlags::CGEventFlagCommand),
    (63, CGEventFlags::CGEventFlagSecondaryFn),
];

pub fn valid_name(name: &str) -> bool {
    KEYCODES.iter().any(|(_, n)| *n == name)
}

fn key_for(code: u16) -> Option<Key> {
    let (_, name) = KEYCODES.iter().find(|(c, _)| *c == code)?;
    Key::from_name(name).map(canonical)
}

fn code_for(k: Key) -> Result<u16> {
    let name = match k.name() {
        // the config stores canonical (left) modifiers
        n => n,
    };
    KEYCODES.iter().find(|(_, n)| *n == name).map(|(c, _)| *c).with_context(|| format!("{name} is not a key on a Mac keyboard"))
}

fn flag_for(k: Key) -> Option<CGEventFlags> {
    Some(match k.name() {
        "KEY_LEFTCTRL" => CGEventFlags::CGEventFlagControl,
        "KEY_LEFTSHIFT" => CGEventFlags::CGEventFlagShift,
        "KEY_LEFTALT" | "KEY_RIGHTALT" => CGEventFlags::CGEventFlagAlternate,
        "KEY_LEFTMETA" => CGEventFlags::CGEventFlagCommand,
        _ => return None,
    })
}

/// Key down/up events (canonicalised, repeats dropped) from every keyboard.
pub fn spawn_key_listener() -> Result<Receiver<(Key, bool)>> {
    // Asks macOS to show its permission prompt the first time; a no-op once decided.
    unsafe {
        if !CGPreflightListenEventAccess() {
            CGRequestListenEventAccess();
        }
    }
    let (tx, rx) = channel::<(Key, bool)>();
    let (ready_tx, ready_rx) = channel::<bool>();
    std::thread::spawn(move || {
        // Modifier keys come as FlagsChanged without saying which way: track which are down.
        let held: Mutex<HashSet<u16>> = Mutex::new(HashSet::new());
        let tx2 = tx.clone();
        let result = CGEventTap::with_enabled(
            CGEventTapLocation::HID,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::ListenOnly,
            vec![CGEventType::KeyDown, CGEventType::KeyUp, CGEventType::FlagsChanged],
            move |_proxy, kind, event| {
                let code = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE) as u16;
                match kind {
                    CGEventType::KeyDown | CGEventType::KeyUp => {
                        let down = matches!(kind, CGEventType::KeyDown);
                        if down && event.get_integer_value_field(EventField::KEYBOARD_EVENT_AUTOREPEAT) != 0 {
                            return CallbackResult::Keep;
                        }
                        if let Some(k) = key_for(code) {
                            let _ = tx2.send((k, down));
                        }
                    }
                    CGEventType::FlagsChanged => {
                        let Some((_, flag)) = MODIFIERS.iter().find(|(c, _)| *c == code) else {
                            return CallbackResult::Keep;
                        };
                        let mut held = held.lock().unwrap();
                        let down = if !event.get_flags().contains(*flag) {
                            // no key of this kind is down any more (left and right share a flag)
                            held.retain(|c| MODIFIERS.iter().find(|(m, _)| m == c).is_none_or(|(_, f)| f != flag));
                            false
                        } else {
                            held.insert(code)
                        };
                        if let Some(k) = key_for(code) {
                            let _ = tx2.send((k, down));
                        }
                    }
                    _ => {}
                }
                CallbackResult::Keep
            },
            || {
                let _ = ready_tx.send(true);
                CFRunLoop::run_current();
            },
        );
        if result.is_err() {
            let _ = ready_tx.send(false);
        }
    });
    match ready_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(true) => Ok(rx),
        _ => bail!("{PERMISSION_HINT}"),
    }
}

/// Types text with posted keyboard events carrying Unicode strings, so any character works whatever the
/// keyboard layout.
pub struct Typer {
    source: CGEventSource,
}

impl Typer {
    pub fn new() -> Result<Self> {
        if !unsafe { AXIsProcessTrusted() } {
            eprintln!(
                "warning: typing into other windows needs Accessibility access: allow your terminal under \
                 System Settings > Privacy & Security > Accessibility (until then keystrokes are dropped; \
                 --output clipboard works without it)"
            );
        }
        let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|_| anyhow::anyhow!("cannot create a keyboard event source"))?;
        Ok(Self { source })
    }

    /// Anything can be typed: macOS takes the characters as Unicode, not as keys.
    pub fn can_type(&self, _c: char) -> bool {
        true
    }

    fn post(&self, code: u16, down: bool, flags: CGEventFlags, text: Option<&str>) -> Result<()> {
        let ev = CGEvent::new_keyboard_event(self.source.clone(), code, down)
            .map_err(|_| anyhow::anyhow!("cannot create a key event"))?;
        if let Some(t) = text {
            ev.set_string(t);
        }
        ev.set_flags(flags);
        ev.post(CGEventTapLocation::HID);
        Ok(())
    }

    pub fn type_text(&mut self, text: &str) -> Result<()> {
        // A key event carries at most 20 UTF-16 units of text; Return and Tab are real keys.
        let mut chunk = String::new();
        let mut units = 0;
        let flush = |this: &Self, chunk: &mut String, units: &mut usize| -> Result<()> {
            if !chunk.is_empty() {
                this.post(0, true, CGEventFlags::CGEventFlagNull, Some(chunk))?;
                this.post(0, false, CGEventFlags::CGEventFlagNull, Some(chunk))?;
                std::thread::sleep(Duration::from_millis(4));
                chunk.clear();
                *units = 0;
            }
            Ok(())
        };
        for c in text.chars() {
            match c {
                '\n' | '\t' => {
                    flush(self, &mut chunk, &mut units)?;
                    let code = if c == '\n' { 36 } else { 48 };
                    self.post(code, true, CGEventFlags::CGEventFlagNull, None)?;
                    self.post(code, false, CGEventFlags::CGEventFlagNull, None)?;
                    std::thread::sleep(Duration::from_millis(4));
                }
                c => {
                    if units + c.len_utf16() > 20 {
                        flush(self, &mut chunk, &mut units)?;
                    }
                    chunk.push(c);
                    units += c.len_utf16();
                }
            }
        }
        flush(self, &mut chunk, &mut units)
    }

    /// Presses a shortcut such as Cmd+V.
    pub fn press_combo(&mut self, keys: &BTreeSet<Key>) -> Result<()> {
        let (mods, main): (Vec<Key>, Vec<Key>) = keys.iter().partition(|k| is_modifier(**k));
        let main = *main.first().context("a paste shortcut needs a non-modifier key")?;
        let flags = mods.iter().filter_map(|m| flag_for(*m)).fold(CGEventFlags::CGEventFlagNull, |a, b| a | b);
        let code = code_for(main)?;
        self.post(code, true, flags, None)?;
        self.post(code, false, flags, None)?;
        std::thread::sleep(Duration::from_millis(4));
        Ok(())
    }
}

pub fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut child = std::process::Command::new("pbcopy").stdin(std::process::Stdio::piped()).spawn().context("pbcopy not found")?;
    child.stdin.take().unwrap().write_all(text.as_bytes())?;
    child.wait()?;
    Ok(())
}

/// Notification Center banners through `osascript`.
pub struct Notifier {
    enabled: bool,
}

impl Notifier {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn show(&mut self, summary: &str, body: &str, _ms: u32) {
        if !self.enabled {
            return;
        }
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let script = format!(r#"display notification "{}" with title "Fluid dictation" subtitle "{}""#, esc(body), esc(summary));
        let _ = std::process::Command::new("osascript").args(["-e", &script]).output();
    }
}
