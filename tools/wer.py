"""WER of a hyps JSONL ({"id","hyp"}) against a manifest JSONL ({"id","text"}), with simple normalisation."""
import json
import re
import sys


def norm(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"'(?!\w)|(?<!\w)'", " ", s)
    return s.split()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)]


refs = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(sys.argv[1])}
hyps = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in open(sys.argv[2])}
E = N = 0
for k, h in hyps.items():
    r = norm(refs[k])
    E += edits(r, norm(h))
    N += len(r)
print(f"{len(hyps)} utts  WER {100 * E / N:.2f}%  ({E}/{N})")
