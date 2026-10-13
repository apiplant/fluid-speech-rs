"""Run the Core ML package end-to-end with the numpy MIL interpreter (one 15 s window).

usage: coreml_ref.py <coreml dir> <audio.wav> [--dump out.npz]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import mil  # noqa: E402

WINDOW = 240000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("audio")
    ap.add_argument("--dump")
    args = ap.parse_args()

    wav, sr = sf.read(args.audio, dtype="float32")
    assert sr == 16000 and wav.ndim == 1
    wav = wav[:WINDOW]
    n = len(wav)
    audio = np.zeros((1, WINDOW), np.float32)
    audio[0, :n] = wav

    t = time.time()
    pre = mil.Model(os.path.join(args.model_dir, "Preprocessor.mlmodelc"))
    out = pre.run({"audio_signal": audio, "audio_length": np.array([n], np.int32)})
    mel, mel_len = out["mel"], out["mel_length"]
    print("mel", mel.shape, mel_len, f"{time.time() - t:.1f}s", file=sys.stderr)

    t = time.time()
    enc = mil.Model(os.path.join(args.model_dir, "Encoder.mlmodelc"))
    out = enc.run({"mel": mel, "mel_length": mel_len})
    encoder, enc_len = out["encoder"], int(out["encoder_length"][0])
    print("encoder", encoder.shape, enc_len, f"{time.time() - t:.1f}s", file=sys.stderr)

    dec = mil.Model(os.path.join(args.model_dir, "Decoder.mlmodelc"))
    joint = mil.Model(os.path.join(args.model_dir, "JointDecision.mlmodelc"))
    vocab = json.load(open(os.path.join(args.model_dir, "parakeet_vocab.json")))
    blank, durations = 8192, [0, 1, 2, 3, 4]

    h = np.zeros((2, 1, 640), np.float32)
    c = np.zeros((2, 1, 640), np.float32)

    def run_dec(tok, h, c):
        o = dec.run({"targets": np.array([[tok]], np.int32), "target_length": np.array([1], np.int32), "h_in": h, "c_in": c})
        return o["decoder"], o["h_out"], o["c_out"]

    d, h, c = run_dec(blank, h, c)
    tokens, t, emitted = [], 0, 0
    while t < enc_len:
        o = joint.run({"encoder_step": encoder[:, :, t:t + 1], "decoder_step": d})
        tok, dur = int(o["token_id"].ravel()[0]), durations[int(o["duration"].ravel()[0])]
        if tok != blank:
            tokens.append(tok)
            d, h, c = run_dec(tok, h, c)
            emitted += 1
        if tok == blank and dur == 0:
            dur = 1
        if dur == 0 and emitted >= 10:
            dur, emitted = 1, 0
        if dur:
            emitted = 0
        t += dur
    text = "".join(vocab[str(i)] for i in tokens).replace("▁", " ").strip()
    print(text)
    print(tokens, file=sys.stderr)
    if args.dump:
        np.savez(args.dump, mel=mel, mel_length=mel_len, encoder=encoder, encoder_length=enc_len, tokens=np.array(tokens))


if __name__ == "__main__":
    main()
