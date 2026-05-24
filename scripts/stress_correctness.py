#!/usr/bin/env python3
"""Stress-correctness probe: hammer an OpenAI-compatible serve with many varied
requests and detect when (if) output degrades into garbage (token-0 '!' collapse
or other degenerate repetition). Used to reliably reproduce the compiled-path
corruption and to A/B test mitigations.

Reports: first bad request index, total bad, and a few samples. Stdlib only.
Usage: stress_correctness.py --base-url URL --model M [--n 60] [--label L]
"""
import argparse, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "What is the capital of France? One word.",
    "What is 6 times 7? Number only.",
    "Name a primary color. One word.",
    "What is the capital of Japan? One word.",
    "What is 100 minus 37? Number only.",
    "What is the largest planet? One word.",
    "Spell the word 'cat'.",
    "What comes after Wednesday? One word.",
    "Write one short sentence about the sun.",
    "List three fruits, comma separated.",
]

def chat(base, model, content, max_tokens=120, timeout=180):
    p = {"model": model, "messages": [{"role":"user","content":content}],
         "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(),
                                 headers={"Content-Type":"application/json"}, method="POST")
    r = urllib.request.urlopen(req, timeout=timeout)
    d = json.loads(r.read().decode())
    return d["choices"][0]["message"]["content"], d["choices"][0]["finish_reason"]

def is_garbage(text):
    if not text.strip():
        return True
    # token-0 '!' collapse
    if text.count("!") > 25:
        return True
    # degenerate repetition: very low unique-word ratio over enough words
    w = text.split()
    if len(w) > 40 and len(set(w))/len(w) < 0.15:
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--label", default="")
    ap.add_argument("--batch", action="store_true", help="also interleave concurrency-4 bursts")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    print(f"=== stress_correctness {a.label} : {a.n} reqs @ {base} ===", flush=True)
    first_bad = None; nbad = 0; samples = []
    t0 = time.time()
    for i in range(a.n):
        content = PROMPTS[i % len(PROMPTS)]
        # vary max_tokens to exercise different shapes
        mt = 80 + (i % 5) * 80
        try:
            out, fin = chat(base, a.model, content, max_tokens=mt)
        except Exception as e:
            out, fin = f"<ERR {e!r}>", "error"
        bad = is_garbage(out)
        if bad:
            nbad += 1
            if first_bad is None:
                first_bad = i
            if len(samples) < 3:
                samples.append((i, content, out[-50:]))
        flag = "GARBAGE" if bad else "ok"
        if bad or i % 10 == 0:
            print(f"  req{i:3d} mt={mt} [{flag}] {out[-45:]!r}", flush=True)
        # occasional concurrency-4 burst
        if a.batch and i % 13 == 12:
            with ThreadPoolExecutor(max_workers=4) as ex:
                outs = list(ex.map(lambda c: chat(base, a.model, c, 100)[0], PROMPTS[:4]))
            for o in outs:
                if is_garbage(o):
                    nbad += 1
                    if first_bad is None: first_bad = i
    dur = time.time()-t0
    print(f"\n=== RESULT {a.label}: {nbad} garbage / {a.n} reqs; first_bad_at={first_bad}; {dur:.0f}s ===", flush=True)
    for idx, q, tail in samples:
        print(f"   sample req{idx}: {q!r} -> ...{tail!r}", flush=True)
    print("VERDICT:", "CLEAN" if nbad == 0 else "CORRUPTS", flush=True)

if __name__ == "__main__":
    main()
