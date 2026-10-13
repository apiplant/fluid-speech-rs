//! Words with timestamps and confidences, plain-text joining and SRT subtitles.

use crate::vocab::Vocab;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct Word {
    pub word: String,
    /// seconds
    pub start: f32,
    pub end: f32,
    /// geometric mean of the word's token probabilities
    pub confidence: f32,
}

/// Group tokens `(id, time in seconds, probability)` into words. A word ends where the next begins (or one frame
/// after its last token).
pub fn words(vocab: &Vocab, tokens: &[(u32, f32, f32)], frame_secs: f32) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut logp: Vec<(f32, u32)> = Vec::new(); // per word: sum of ln p, token count
    let mut last_t = 0f32;
    for &(id, t, p) in tokens {
        if vocab.is_special(id) {
            continue;
        }
        let piece = vocab.piece(id);
        // word boundaries are marked by a leading space (or SentencePiece's U+2581)
        let (new_word, body) = match piece.strip_prefix([' ', '\u{2581}']) {
            Some(rest) => (true, rest),
            None => (words.is_empty(), piece),
        };
        if new_word {
            if let Some(w) = words.last_mut() {
                w.end = t.max(w.start);
            }
            words.push(Word { word: body.to_string(), start: t, end: t, confidence: 0.0 });
            logp.push((0.0, 0));
        } else if let Some(w) = words.last_mut() {
            w.word.push_str(body);
        }
        let l = logp.last_mut().unwrap();
        l.0 += p.max(1e-9).ln();
        l.1 += 1;
        last_t = t;
    }
    if let Some(w) = words.last_mut() {
        w.end = last_t + frame_secs;
    }
    for (w, (sum, n)) in words.iter_mut().zip(&logp) {
        w.confidence = (sum / (*n).max(1) as f32).exp();
    }
    words.retain(|w| !w.word.is_empty());
    words
}

/// Geometric mean of the word confidences (0 for no words).
pub fn confidence(words: &[Word]) -> f32 {
    if words.is_empty() {
        return 0.0;
    }
    (words.iter().map(|w| w.confidence.max(1e-9).ln()).sum::<f32>() / words.len() as f32).exp()
}

pub fn join(words: &[Word]) -> String {
    words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ")
}

pub fn srt_time(t: f32) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

pub fn srt(words: &[Word]) -> String {
    // cues end at sentence punctuation, a pause over 1 s, or ~7 s / 16 words
    let mut out = String::new();
    let mut cue: Vec<&Word> = Vec::new();
    let mut n = 0;
    let mut flush = |cue: &mut Vec<&Word>, out: &mut String| {
        if let (Some(a), Some(b)) = (cue.first(), cue.last()) {
            n += 1;
            let text = cue.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
            out.push_str(&format!("{n}\n{} --> {}\n{text}\n\n", srt_time(a.start), srt_time(b.end)));
        }
        cue.clear();
    };
    for (i, w) in words.iter().enumerate() {
        cue.push(w);
        let next_gap = words.get(i + 1).map(|nx| nx.start - w.end).unwrap_or(0.0);
        let span = w.end - cue[0].start;
        if w.word.ends_with(['.', '?', '!']) || next_gap > 1.0 || span > 7.0 || cue.len() >= 16 {
            flush(&mut cue, &mut out);
        }
    }
    flush(&mut cue, &mut out);
    out
}
