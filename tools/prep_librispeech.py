"""Download LibriSpeech test-clean, convert to 16 kHz WAV and write data/ls-test-clean.jsonl ({id, audio, text})."""
import glob
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf

os.makedirs("data", exist_ok=True)
if not os.path.isdir("data/LibriSpeech/test-clean"):
    subprocess.run("curl -sL https://www.openslr.org/resources/12/test-clean.tar.gz | tar xz -C data", shell=True, check=True)
root, out = "data/LibriSpeech/test-clean", "data/ls-test-clean-wav"
os.makedirs(out, exist_ok=True)
refs = {}
for t in glob.glob(root + "/*/*/*.trans.txt"):
    for line in open(t):
        k, txt = line.strip().split(" ", 1)
        refs[k] = txt


def conv(k):
    spk, ch, _ = k.split("-")
    a, sr = sf.read(f"{root}/{spk}/{ch}/{k}.flac", dtype="int16")
    sf.write(f"{out}/{k}.wav", a, sr, subtype="PCM_16")


with ThreadPoolExecutor(32) as ex:
    list(ex.map(conv, sorted(refs)))
with open("data/ls-test-clean.jsonl", "w") as f:
    for k in sorted(refs):
        f.write(json.dumps({"id": k, "audio": f"{out}/{k}.wav", "text": refs[k]}) + "\n")
print(len(refs), "utterances")
