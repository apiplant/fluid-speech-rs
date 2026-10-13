//! Platform-independent keys for the dictation hotkey and for typing.
//!
//! A [`Key`] is named like a Linux evdev key (`KEY_LEFTCTRL`, `KEY_SPACE`, `KEY_F9`, ...): that is what the
//! config file stores, on every platform, so a hotkey saved on Linux means the same thing on macOS
//! (`KEY_LEFTMETA` is Super on Linux and Command on macOS). Each platform backend maps between its own key
//! events and these names.

use anyhow::{Result, bail};
use std::collections::BTreeSet;
use std::sync::Mutex;

use super::platform;

/// A key, identified by its evdev-style name.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(&'static str);

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

static INTERNED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

impl Key {
    /// A key with a fixed, known name.
    pub const fn new(name: &'static str) -> Key {
        Key(name)
    }

    pub fn name(self) -> &'static str {
        self.0
    }

    /// The key called `name` (case-insensitive, `KEY_`/`BTN_` prefixed), if this platform knows it.
    pub fn from_name(name: &str) -> Option<Key> {
        let upper = name.to_uppercase();
        if !platform::valid_name(&upper) {
            return None;
        }
        let mut seen = INTERNED.lock().unwrap();
        if let Some(found) = seen.iter().find(|n| **n == upper) {
            return Some(Key(found));
        }
        // Names are few and live for the whole process: leaking one `str` per distinct key is the simplest
        // way to hand out `Copy` keys.
        let leaked: &'static str = Box::leak(upper.into_boxed_str());
        seen.push(leaked);
        Some(Key(leaked))
    }
}

pub const LEFTCTRL: Key = Key::new("KEY_LEFTCTRL");
pub const LEFTSHIFT: Key = Key::new("KEY_LEFTSHIFT");
pub const LEFTALT: Key = Key::new("KEY_LEFTALT");
pub const RIGHTALT: Key = Key::new("KEY_RIGHTALT");
pub const LEFTMETA: Key = Key::new("KEY_LEFTMETA");
pub const ENTER: Key = Key::new("KEY_ENTER");
pub const TAB: Key = Key::new("KEY_TAB");
pub const BACKSPACE: Key = Key::new("KEY_BACKSPACE");
pub const DELETE: Key = Key::new("KEY_DELETE");

/// Left and right Ctrl / Shift / Meta count as the same key, so a combo saved with one side matches either.
pub fn canonical(k: Key) -> Key {
    match k.0 {
        "KEY_RIGHTCTRL" => LEFTCTRL,
        "KEY_RIGHTSHIFT" => LEFTSHIFT,
        "KEY_RIGHTMETA" => LEFTMETA,
        _ => k,
    }
}

pub fn is_modifier(k: Key) -> bool {
    matches!(canonical(k), LEFTCTRL | LEFTSHIFT | LEFTMETA | LEFTALT | RIGHTALT)
}

/// "ctrl+alt+d", "super+space", "cmd+shift+d", "F9", "KEY_PAUSE" -> canonical keys
pub fn parse_combo(s: &str) -> Result<BTreeSet<Key>> {
    let mut out = BTreeSet::new();
    for tok in s.split('+').map(str::trim).filter(|t| !t.is_empty()) {
        let name = match tok.to_lowercase().as_str() {
            "ctrl" | "control" => "KEY_LEFTCTRL".to_string(),
            "shift" => "KEY_LEFTSHIFT".to_string(),
            "alt" | "option" | "opt" => "KEY_LEFTALT".to_string(),
            "altgr" => "KEY_RIGHTALT".to_string(),
            "super" | "meta" | "win" | "cmd" | "command" | "logo" => "KEY_LEFTMETA".to_string(),
            "esc" => "KEY_ESC".to_string(),
            "del" => "KEY_DELETE".to_string(),
            "return" => "KEY_ENTER".to_string(),
            _ if tok.to_uppercase().starts_with("KEY_") || tok.to_uppercase().starts_with("BTN_") => tok.to_uppercase(),
            t => format!("KEY_{}", t.to_uppercase()),
        };
        let k = Key::from_name(&name).ok_or_else(|| anyhow::anyhow!("unknown key {tok:?} ({name})"))?;
        out.insert(canonical(k));
    }
    if out.is_empty() {
        bail!("empty key combination");
    }
    Ok(out)
}

/// A human-readable name for a combo, modifiers first. The meta key is "Cmd" on macOS and "Super" elsewhere.
pub fn combo_name(keys: &BTreeSet<Key>) -> String {
    let mut v: Vec<&Key> = keys.iter().collect();
    v.sort_by_key(|k| !is_modifier(**k));
    v.iter()
        .map(|k| {
            let s = k.0.strip_prefix("KEY_").unwrap_or(k.0);
            match s {
                "LEFTCTRL" => "Ctrl".into(),
                "LEFTSHIFT" => "Shift".into(),
                "LEFTALT" if cfg!(target_os = "macos") => "Option".into(),
                "LEFTALT" => "Alt".into(),
                "RIGHTALT" if cfg!(target_os = "macos") => "Right Option".into(),
                "RIGHTALT" => "AltGr".into(),
                "LEFTMETA" if cfg!(target_os = "macos") => "Cmd".into(),
                "LEFTMETA" => "Super".into(),
                _ => s.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join("+")
}

/// US-layout key for a character: (key, needs shift)
pub fn us_key(c: char) -> Option<(Key, bool)> {
    const PUNCT: &[(char, &str, bool)] = &[
        ('!', "KEY_1", true), ('@', "KEY_2", true), ('#', "KEY_3", true), ('$', "KEY_4", true), ('%', "KEY_5", true),
        ('^', "KEY_6", true), ('&', "KEY_7", true), ('*', "KEY_8", true), ('(', "KEY_9", true), (')', "KEY_0", true),
        ('-', "KEY_MINUS", false), ('_', "KEY_MINUS", true), ('=', "KEY_EQUAL", false), ('+', "KEY_EQUAL", true),
        ('[', "KEY_LEFTBRACE", false), ('{', "KEY_LEFTBRACE", true), (']', "KEY_RIGHTBRACE", false),
        ('}', "KEY_RIGHTBRACE", true), ('\\', "KEY_BACKSLASH", false), ('|', "KEY_BACKSLASH", true),
        (';', "KEY_SEMICOLON", false), (':', "KEY_SEMICOLON", true), ('\'', "KEY_APOSTROPHE", false),
        ('"', "KEY_APOSTROPHE", true), ('`', "KEY_GRAVE", false), ('~', "KEY_GRAVE", true), (',', "KEY_COMMA", false),
        ('<', "KEY_COMMA", true), ('.', "KEY_DOT", false), ('>', "KEY_DOT", true), ('/', "KEY_SLASH", false),
        ('?', "KEY_SLASH", true),
    ];
    const LETTERS: [&str; 26] = [
        "KEY_A", "KEY_B", "KEY_C", "KEY_D", "KEY_E", "KEY_F", "KEY_G", "KEY_H", "KEY_I", "KEY_J", "KEY_K", "KEY_L",
        "KEY_M", "KEY_N", "KEY_O", "KEY_P", "KEY_Q", "KEY_R", "KEY_S", "KEY_T", "KEY_U", "KEY_V", "KEY_W", "KEY_X",
        "KEY_Y", "KEY_Z",
    ];
    const DIGITS: [&str; 10] =
        ["KEY_0", "KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5", "KEY_6", "KEY_7", "KEY_8", "KEY_9"];
    Some(match c {
        'a'..='z' => (Key::new(LETTERS[c as usize - 'a' as usize]), false),
        'A'..='Z' => (Key::new(LETTERS[c as usize - 'A' as usize]), true),
        '0'..='9' => (Key::new(DIGITS[c as usize - '0' as usize]), false),
        ' ' => (Key::new("KEY_SPACE"), false),
        '\n' => (ENTER, false),
        '\t' => (TAB, false),
        _ => {
            let (_, name, shift) = PUNCT.iter().find(|(ch, _, _)| *ch == c)?;
            (Key::new(name), *shift)
        }
    })
}

/// Typographic characters folded to what a US keyboard can type.
pub fn plain(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' => "'".to_string(),
            '\u{201C}' | '\u{201D}' | '\u{201F}' | '\u{2033}' => "\"".to_string(),
            '\u{2013}' | '\u{2014}' | '\u{2212}' => "-".to_string(),
            '\u{2026}' => "...".to_string(),
            '\u{00A0}' | '\u{202F}' => " ".to_string(),
            c => c.to_string(),
        })
        .collect()
}

/// Keys that produce text (and auto-repeat) in the focused window when pressed without Ctrl/Alt/Super.
pub fn types_text(k: Key) -> bool {
    (32u8..127).filter_map(|c| us_key(c as char)).any(|(t, _)| t == k) || matches!(k, ENTER | TAB | BACKSPACE | DELETE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combos_parse_and_canonicalise() {
        let c = parse_combo("ctrl+alt+d").unwrap();
        assert!(c.contains(&LEFTCTRL) && c.contains(&LEFTALT) && c.contains(&Key::new("KEY_D")));
        assert_eq!(parse_combo("cmd+shift+space").unwrap(), parse_combo("super+shift+space").unwrap());
        assert!(parse_combo("").is_err());
        assert!(parse_combo("ctrl+notakey").is_err());
    }

    #[test]
    fn us_keys_cover_ascii() {
        assert_eq!(us_key('A'), Some((Key::new("KEY_A"), true)));
        assert_eq!(us_key('?'), Some((Key::new("KEY_SLASH"), true)));
        assert_eq!(us_key('é'), None);
    }
}
