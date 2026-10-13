//! Platform layers for `fluid-dictate`: keys, hotkey capture, typing, clipboard and notifications.
//!
//! Linux reads /dev/input and types through /dev/uinput; macOS watches a Quartz event tap and posts key events.
//! Both expose the same functions and types, re-exported from [`platform`].

pub mod key;

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "linux")]
pub use linux as platform;

#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "macos")]
pub use macos as platform;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("fluid-dictate supports Linux and macOS");
