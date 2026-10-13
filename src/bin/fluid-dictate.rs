//! Push-button dictation for Linux (Wayland or X11) and macOS: a hotkey starts and stops recording from the
//! microphone, and the transcript is typed into the focused window.
//!
//! On Linux keys are read straight from /dev/input (evdev) and text is typed through /dev/uinput, so it works
//! under any compositor; both need the user in the `input` group (or equivalent udev rules). On macOS the keys
//! come from a Quartz event tap and the text from posted key events; both need privacy permissions (Input
//! Monitoring and Accessibility) for the terminal that runs it. See `fluid_speech::dictate`.

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use fluid_speech::dictate::key::{Key, combo_name, is_modifier, parse_combo, plain, types_text};
use fluid_speech::dictate::platform::{self, Notifier, Typer, copy_to_clipboard, spawn_key_listener};
use fluid_speech::{audio, mic, text, DeviceArg, Engine, LoadOptions};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

/// Dictation with Fluid Speech Mini: press a hotkey, speak, press it again; the text is typed where your cursor is.
#[derive(Parser)]
#[command(name = "fluid-dictate", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand)]
enum Cmd {
    /// Press the key or key combination to use as the hotkey; it is saved to the config file
    Setup {
        /// Push-to-talk: record while the keys are held instead of toggling on each press
        #[arg(long)]
        hold: bool,
    },
    /// Start or stop recording in the running daemon (bind this in your desktop's shortcut settings instead of
    /// using the built-in hotkey)
    Toggle,
    /// Start recording in the running daemon
    Start,
    /// Stop recording in the running daemon and type the transcript
    Stop,
    /// Print where the config file lives and what it holds
    Config,
}

#[derive(Args)]
struct RunArgs {
    /// Hotkey, overriding the saved one: e.g. "F9", "ctrl+alt+d", "super+space" ("cmd+shift+d" on a Mac), "KEY_PAUSE"
    #[arg(long)]
    key: Option<String>,

    /// Push-to-talk: record only while the hotkey is held
    #[arg(long)]
    hold: bool,

    /// No hotkey at all: only the control socket (`fluid-dictate toggle`) starts and stops recording
    #[arg(long, conflicts_with = "key")]
    no_hotkey: bool,

    /// Microphone: any part of a name from `fluid-speech --list-mics` (default: the system default input)
    #[arg(long)]
    mic_device: Option<String>,

    /// How the transcript is delivered
    #[arg(long, value_enum, default_value_t = Output::Type)]
    output: Output,

    /// Shortcut pressed to paste (for --output paste, and when typing falls back to paste for characters like
    /// "è" that a US keymap can't type). Terminals on Linux usually want "ctrl+shift+v"
    #[arg(long, default_value = platform::DEFAULT_PASTE_KEYS)]
    paste_keys: String,

    /// Don't add a space after each dictation
    #[arg(long)]
    no_space: bool,

    /// No desktop notifications
    #[arg(long)]
    quiet: bool,

    /// Stop recording automatically after this many seconds
    #[arg(long, default_value_t = 300.0)]
    max_secs: f32,

    /// Model directory (model.safetensors, config.json, vocab.json). Default: ./model, ../model, next to the
    /// binary, else downloaded from Hugging Face into ~/.cache/fluid-speech-rs on first use
    #[arg(short, long, env = "FLUID_MODEL")]
    model: Option<PathBuf>,

    #[arg(short, long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// CUDA: fp16 accumulation in encoder GEMMs (faster, WER within noise)
    #[arg(long)]
    fast: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Output {
    /// Type it (a virtual US keyboard on Linux, any character on macOS); falls back to paste for characters it
    /// cannot type
    Type,
    /// Put the text on the clipboard (wl-copy, pbcopy) and press the paste shortcut
    Paste,
    /// Only put the text on the clipboard
    Clipboard,
    /// Print to stdout
    Stdout,
}

// ---------------------------------------------------------------------------------------------------------------
// config

#[derive(Serialize, Deserialize, Default)]
struct Config {
    /// evdev-style key names (on every platform), e.g. ["KEY_LEFTCTRL", "KEY_SPACE"]; KEY_LEFTMETA is Super or Cmd
    keys: Vec<String>,
    #[serde(default)]
    hold: bool,
}

fn config_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config"));
    base.join("fluid-speech").join("dictate.json")
}

fn load_config() -> Result<Option<Config>> {
    let p = config_path();
    match std::fs::read(&p) {
        Ok(b) => Ok(Some(serde_json::from_slice(&b).with_context(|| format!("parsing {}", p.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", p.display())),
    }
}

fn save_config(c: &Config) -> Result<PathBuf> {
    let p = config_path();
    std::fs::create_dir_all(p.parent().unwrap())?;
    std::fs::write(&p, serde_json::to_string_pretty(c)? + "\n")?;
    Ok(p)
}

fn socket_path() -> PathBuf {
    let dir = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    dir.join("fluid-dictate.sock")
}

// ---------------------------------------------------------------------------------------------------------------
// setup

fn setup(hold: bool) -> Result<()> {
    let rx = spawn_key_listener()?;
    eprintln!("Press the key or key combination to use for dictation, then release it.");
    eprintln!("Tip: pick something that does nothing else (F9-F12, Pause, Super+<letter> you don't use, a mouse side button):");
    eprintln!("     the keys also reach the focused window.");
    // the Enter that started this command is still being released: drain it
    std::thread::sleep(Duration::from_millis(300));
    while rx.try_recv().is_ok() {}

    // capture twice so a stray key (or someone else typing) can't end up as the hotkey
    let chord = loop {
        let first = read_chord(&rx)?;
        eprintln!("Got {}. Press it again to confirm.", combo_name(&first));
        let second = read_chord(&rx)?;
        if second == first {
            break first;
        }
        eprintln!("That was {}, not {}. Start over: press the combination.", combo_name(&second), combo_name(&first));
    };
    // keys that produce text (and auto-repeat) in the focused window when not combined with Ctrl/Alt/Super
    let has_mod = chord.iter().any(|k| is_modifier(*k) && k.name() != "KEY_LEFTSHIFT");
    if !has_mod && chord.iter().any(|k| types_text(*k)) {
        eprintln!(
            "Warning: {} includes a key that types into the focused window (and auto-repeats while held). \
             Consider adding Ctrl/Alt/Super (Cmd/Option on a Mac), or use F13-F24 / Pause.",
            combo_name(&chord)
        );
    }
    if chord.iter().all(|k| is_modifier(*k)) && chord.len() == 1 {
        eprintln!("Note: a lone modifier as the hotkey will also fire whenever you use it in other shortcuts.");
    }
    let cfg = Config { keys: chord.iter().map(|k| format!("{k:?}")).collect(), hold };
    let path = save_config(&cfg)?;
    eprintln!(
        "Saved {} ({}) to {}",
        combo_name(&chord),
        if hold { "hold to talk" } else { "press to start, press again to stop" },
        path.display()
    );
    eprintln!("Now run `fluid-dictate` (or restart it if it is already running).");
    Ok(())
}

/// Every key held together from the first press until all are released.
fn read_chord(rx: &Receiver<(Key, bool)>) -> Result<BTreeSet<Key>> {
    let mut down: HashSet<Key> = HashSet::new();
    let mut chord: BTreeSet<Key> = BTreeSet::new();
    loop {
        let (k, pressed) = rx.recv()?;
        if pressed {
            down.insert(k);
            chord.insert(k);
        } else if down.remove(&k) && down.is_empty() && !chord.is_empty() {
            return Ok(chord);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// daemon

enum Event {
    Key(Key, bool),
    Command(String, UnixStream),
}

fn send_command(cmd: &str) -> Result<()> {
    let path = socket_path();
    let mut s = UnixStream::connect(&path)
        .with_context(|| format!("no fluid-dictate daemon listening on {}", path.display()))?;
    writeln!(s, "{cmd}")?;
    let mut reply = String::new();
    BufReader::new(s).read_line(&mut reply)?;
    eprint!("{reply}");
    Ok(())
}

fn spawn_socket(tx: Sender<Event>) -> Result<()> {
    let path = socket_path();
    if UnixStream::connect(&path).is_ok() {
        bail!("fluid-dictate is already running ({})", path.display());
    }
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).with_context(|| format!("binding {}", path.display()))?;
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut line = String::new();
            if BufReader::new(&stream).read_line(&mut line).is_ok() {
                let _ = tx.send(Event::Command(line.trim().to_string(), stream));
            }
        }
    });
    Ok(())
}

struct Recording {
    capture: mic::Capture,
    started: Instant,
}

struct Daemon {
    engine: Engine,
    args: RunArgs,
    keyboard: Option<Typer>,
    notifier: Notifier,
    down: HashSet<Key>,
    rec: Option<Recording>,
}

impl Daemon {
    fn start(&mut self) -> Result<()> {
        if self.rec.is_some() {
            return Ok(());
        }
        let capture = mic::Capture::open(self.args.mic_device.as_deref())?;
        eprintln!("recording from {}", capture.name);
        self.notifier.show("Listening…", "", 0);
        self.rec = Some(Recording { capture, started: Instant::now() });
        Ok(())
    }

    fn stop(&mut self, rx: &Receiver<Event>) -> Result<()> {
        let Some(rec) = self.rec.take() else { return Ok(()) };
        let rate = rec.capture.rate;
        // keep listening briefly: people press stop while finishing the last word
        let samples = rec.capture.finish(Duration::from_millis(250));
        let secs = samples.len() as f32 / rate as f32;
        if secs < 0.3 {
            self.notifier.show("Too short", "", 1000);
            return Ok(());
        }
        let voiced = mic::voiced_secs(&samples, rate);
        if voiced < mic::MIN_VOICED_SECS {
            eprintln!("{secs:.1}s recorded, {voiced:.2}s of it speech: skipped");
            self.notifier.show("No speech heard", "", 1500);
            return Ok(());
        }
        self.notifier.show("Transcribing…", "", 0);
        let t = Instant::now();
        let audio16 = audio::resample(samples, rate)?;
        let words = self.engine.transcribe(&audio16)?.words;
        if !mic::plausible(&words, 0.9) {
            eprintln!("dropped {:?} (confidence {:.2})", text::join(&words), text::confidence(&words));
            self.notifier.show("No speech heard", "", 1500);
            return Ok(());
        }
        let mut text = plain(&text::join(&words));
        eprintln!("{secs:.1}s of audio transcribed in {:.2?}: {text}", t.elapsed());
        if text.is_empty() {
            self.notifier.show("No speech heard", "", 1500);
            return Ok(());
        }
        if !self.args.no_space {
            text.push(' ');
        }
        self.wait_for_release(rx);
        self.deliver(&text)?;
        self.notifier.show("Dictated", text.trim(), 2500);
        Ok(())
    }

    /// Typing while the hotkey's modifiers are still held would turn letters into shortcuts.
    fn wait_for_release(&mut self, rx: &Receiver<Event>) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while self.down.iter().any(|k| is_modifier(*k)) && Instant::now() < deadline {
            if let Ok(Event::Key(k, pressed)) = rx.recv_timeout(Duration::from_millis(50)) {
                if pressed {
                    self.down.insert(k);
                } else {
                    self.down.remove(&k);
                }
            }
        }
        std::thread::sleep(Duration::from_millis(30));
    }

    fn deliver(&mut self, text: &str) -> Result<()> {
        let typeable = self.keyboard.as_ref().is_some_and(|kb| text.chars().all(|c| kb.can_type(c)));
        match (self.args.output, self.keyboard.as_mut()) {
            (Output::Stdout, _) => {
                println!("{}", text.trim_end());
                Ok(())
            }
            (Output::Type, Some(kb)) if typeable => kb.type_text(text),
            (Output::Type | Output::Paste, Some(kb)) => {
                copy_to_clipboard(text)?;
                std::thread::sleep(Duration::from_millis(100));
                kb.press_combo(&parse_combo(&self.args.paste_keys)?)
            }
            (Output::Clipboard, _) | (Output::Type | Output::Paste, None) => copy_to_clipboard(text),
        }
    }
}

fn run(args: RunArgs) -> Result<()> {
    // hotkey: --key, else the saved one, else none (socket only)
    let cfg = load_config()?;
    let hold = args.hold || cfg.as_ref().is_some_and(|c| c.hold);
    let combo: Option<BTreeSet<Key>> = if args.no_hotkey {
        None
    } else if let Some(k) = &args.key {
        Some(parse_combo(k)?)
    } else if let Some(c) = cfg.as_ref().filter(|c| !c.keys.is_empty()) {
        Some(parse_combo(&c.keys.join("+"))?)
    } else {
        bail!("no hotkey configured: run `fluid-dictate setup`, pass --key, or use --no-hotkey with `fluid-dictate toggle`");
    };

    let (tx, rx) = channel::<Event>();
    spawn_socket(tx.clone())?;

    fluid_speech::init_threads(None);
    let t = Instant::now();
    let opts = LoadOptions { fast_gemm: args.fast, ..Default::default() };
    let engine = Engine::open(args.model.clone(), args.device, &opts)?;
    // CUDA / cuBLAS warm-up, and capture the graphs of typical dictation lengths
    for secs in [1usize, 3, 6, 10] {
        engine.transcribe(&vec![0.0; secs * audio::SAMPLE_RATE])?;
    }
    eprintln!("model ready on {} in {:.1?}", if engine.cuda { "cuda" } else { "cpu" }, t.elapsed());

    let keyboard = match args.output {
        Output::Type | Output::Paste => match Typer::new() {
            Ok(k) => Some(k),
            Err(e) => {
                eprintln!("warning: {e:#}; falling back to the clipboard");
                None
            }
        },
        _ => None,
    };

    if let Some(c) = &combo {
        let keys = spawn_key_listener()?;
        let tx = tx.clone();
        std::thread::spawn(move || {
            for (k, p) in keys {
                if tx.send(Event::Key(k, p)).is_err() {
                    break;
                }
            }
        });
        eprintln!(
            "{}: {}  (or `fluid-dictate toggle`)",
            combo_name(c),
            if hold { "hold to talk" } else { "press to start, press again to stop" }
        );
    } else {
        eprintln!("no hotkey; use `fluid-dictate toggle|start|stop`");
    }

    let notifier = Notifier::new(!args.quiet);
    let max = Duration::from_secs_f32(args.max_secs);
    let mut d = Daemon { engine, args, keyboard, notifier, down: HashSet::new(), rec: None };
    let mut armed = false; // combo fully down, waiting for its release

    loop {
        let ev = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(ev) => Some(ev),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
            Err(_) => break,
        };
        let result = match ev {
            Some(Event::Key(k, pressed)) => {
                // a key can arrive from more than one device node (multi-interface keyboards, remappers):
                // only real transitions count
                let changed = if pressed { d.down.insert(k) } else { d.down.remove(&k) };
                log::debug!("{k:?} {} (changed: {changed}) down: {:?}", if pressed { "down" } else { "up" }, d.down);
                let Some(c) = &combo else { continue };
                if !changed || !c.contains(&k) {
                    Ok(())
                } else if pressed && !armed && d.down.len() == c.len() && c.iter().all(|x| d.down.contains(x)) {
                    armed = true;
                    if hold { d.start() } else if d.rec.is_some() { d.stop(&rx) } else { d.start() }
                } else if !pressed && armed && !c.iter().any(|x| d.down.contains(x)) {
                    // the whole combo is up (not just its first key: Fn keys often report press+release at once)
                    armed = false;
                    if hold { d.stop(&rx) } else { Ok(()) }
                } else {
                    Ok(())
                }
            }
            Some(Event::Command(cmd, mut stream)) => {
                let r = match cmd.as_str() {
                    "toggle" if d.rec.is_some() => d.stop(&rx),
                    "toggle" | "start" => d.start(),
                    "stop" => d.stop(&rx),
                    other => Err(anyhow::anyhow!("unknown command {other:?}")),
                };
                let reply = match &r {
                    Ok(()) => format!("ok: {}\n", if d.rec.is_some() { "recording" } else { "idle" }),
                    Err(e) => format!("error: {e:#}\n"),
                };
                let _ = stream.write_all(reply.as_bytes());
                r
            }
            None => {
                if d.rec.as_ref().is_some_and(|r| r.started.elapsed() > max) {
                    d.stop(&rx)
                } else {
                    Ok(())
                }
            }
        };
        if let Err(e) = result {
            eprintln!("error: {e:#}");
            d.notifier.show("Dictation error", &format!("{e:#}"), 4000);
        }
    }
    Ok(())
}

fn show_config() -> Result<()> {
    let p = config_path();
    match load_config()? {
        Some(c) => {
            let combo = parse_combo(&c.keys.join("+"))?;
            println!("{}: {} ({})", p.display(), combo_name(&combo), if c.hold { "hold" } else { "toggle" });
        }
        None => println!("{}: not set up (run `fluid-dictate setup`)", p.display()),
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,pulseaudio=off")).init();
    let cli = Cli::parse();
    match cli.cmd {
        Some(Cmd::Setup { hold }) => setup(hold),
        Some(Cmd::Toggle) => send_command("toggle"),
        Some(Cmd::Start) => send_command("start"),
        Some(Cmd::Stop) => send_command("stop"),
        Some(Cmd::Config) => show_config(),
        None => run(cli.run),
    }
}
