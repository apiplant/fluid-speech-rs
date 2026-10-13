"""Derived test audio: jfk_{2,4,8}s.wav (prefixes of samples/jfk.wav) and long.wav (~13 min of LibriSpeech
test-clean speakers 1089 + 1188 with short pauses; needs data/ls-test-clean.jsonl, see README)."""
import json
import os

import numpy as np
import soundfile as sf

here = os.path.join(os.path.dirname(__file__), "..")
jfk, sr = sf.read(os.path.join(here, "samples/jfk.wav"), dtype="float32")
for s in (2, 4, 8):
    sf.write(os.path.join(here, f"samples/jfk_{s}s.wav"), jfk[: s * sr], sr, subtype="PCM_16")
manifest = os.path.join(here, "data/ls-test-clean.jsonl")
if os.path.exists(manifest):
    rows = [json.loads(l) for l in open(manifest)]
    rows = [r for r in rows if r["id"].startswith(("1089-134686", "1188-133604"))]
    parts, text = [], []
    for r in rows:
        parts += [sf.read(r["audio"], dtype="float32")[0], np.zeros(4000, np.float32)]
        text.append(r["text"])
    sf.write(os.path.join(here, "samples/long.wav"), np.concatenate(parts), 16000)
    json.dump({"id": "long", "audio": "samples/long.wav", "text": " ".join(text)}, open(os.path.join(here, "samples/long.jsonl"), "w"))
