#!/usr/bin/env python3
"""Version-agnostic functional test suite for a vLLM (OpenAI-compatible) serve.

Tests the features that "it serves + benchmarks" does NOT prove: streaming,
natural EOS, multi-turn, reasoning-token handling, correctness-at-length,
batched-decode correctness, long-context recall, and sampling-param honoring.

Stdlib only (urllib + concurrent.futures). Emits a markdown pass/fail matrix to
stdout and, if --out given, to a file.

Usage:
  feature_test.py --base-url http://127.0.0.1:8080/v1 --model qwen3.6-27b [--out report.md]
"""
import argparse, json, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

def post(base, path, payload, timeout=300, stream=False):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    r = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return r  # caller iterates lines
    return json.loads(r.read().decode())

def chat(base, model, messages, timeout=300, **kw):
    p = {"model": model, "messages": messages, "temperature": 0, **kw}
    return post(base, "/chat/completions", p, timeout=timeout)

RESULTS = []
def record(name, passed, note):
    RESULTS.append((name, passed, note))
    flag = "PASS" if passed else ("WARN" if passed is None else "FAIL")
    print(f"[{flag}] {name}: {note}", flush=True)

def t_basic(base, model):
    # NB: reasoning model -- it emits a <think> block first, so give generous budget.
    try:
        d = chat(base, model, [{"role":"user","content":"Reply with exactly: PONG"}], max_tokens=300, timeout=300)
        out = d["choices"][0]["message"]["content"]
        record("basic_generation", "PONG" in out.upper(), f"finish={d['choices'][0]['finish_reason']} out_tail={out[-60:]!r}")
    except Exception as e:
        record("basic_generation", False, f"ERROR {e!r}")

def t_eos(base, model):
    # A question with a short complete answer should stop NATURALLY (finish=stop), well under the cap.
    try:
        d = chat(base, model, [{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],
                 max_tokens=512, timeout=300)
        fin = d["choices"][0]["finish_reason"]; ct = d["usage"]["completion_tokens"]
        ok = (fin == "stop")
        record("natural_eos", ok, f"finish={fin} (want 'stop', not 'length'), completion_tokens={ct}")
    except Exception as e:
        record("natural_eos", False, f"ERROR {e!r}")

def t_multiturn(base, model):
    try:
        msgs = [
            {"role":"user","content":"My favorite number is 47. Remember it."},
            {"role":"assistant","content":"Got it, your favorite number is 47."},
            {"role":"user","content":"What is my favorite number times 2? Answer with just the number."},
        ]
        d = chat(base, model, msgs, max_tokens=512, timeout=300)
        out = d["choices"][0]["message"]["content"]
        record("multi_turn_context", "94" in out, f"out={out[:120]!r}")
    except Exception as e:
        record("multi_turn_context", False, f"ERROR {e!r}")

def t_streaming(base, model):
    try:
        p = {"model":model,"messages":[{"role":"user","content":"Count from 1 to 5."}],
             "temperature":0,"max_tokens":256,"stream":True}
        r = post(base, "/chat/completions", p, timeout=300, stream=True)
        chunks, got_done, content = 0, False, ""
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": got_done = True; break
            chunks += 1
            try:
                delta = json.loads(data)["choices"][0].get("delta",{}).get("content","")
                if delta: content += delta
            except Exception: pass
        ok = chunks >= 2 and got_done and len(content) > 0
        record("streaming_sse", ok, f"chunks={chunks} done={got_done} len={len(content)}")
    except Exception as e:
        record("streaming_sse", False, f"ERROR {e!r}")

def t_length(base, model):
    # ~300 tokens; check it doesn't collapse into degenerate repetition.
    try:
        d = chat(base, model, [{"role":"user","content":"Write a detailed paragraph about how a CPU executes an instruction."}],
                 max_tokens=320, timeout=400)
        out = d["choices"][0]["message"]["content"]
        words = out.split()
        uniq_ratio = len(set(words))/max(1,len(words))
        # degenerate output repeats heavily -> low unique ratio
        ok = len(words) > 50 and uniq_ratio > 0.30
        record("correctness_at_length", ok, f"words={len(words)} uniq_ratio={uniq_ratio:.2f} (want >0.30)")
    except Exception as e:
        record("correctness_at_length", False, f"ERROR {e!r}")

def t_batched(base, model):
    # 4 concurrent DISTINCT prompts -> each must get its own correct answer (no cross-contamination).
    qa = [("What is the capital of Japan? One word.","tokyo"),
          ("What is the capital of Egypt? One word.","cairo"),
          ("What is the capital of Italy? One word.","rome"),
          ("What is 6 times 7? Just the number.","42")]
    def ask(q):
        d = chat(base, model, [{"role":"user","content":q}], max_tokens=256, timeout=300)
        return d["choices"][0]["message"]["content"]
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            outs = list(ex.map(ask, [q for q,_ in qa]))
        oks = [exp in o.lower() for o,(_,exp) in zip(outs,qa)]
        record("batched_decode_correctness", all(oks),
               f"{sum(oks)}/4 correct: " + " | ".join(f"{exp}:{'Y' if ok else 'N'}" for (_,exp),ok in zip(qa,oks)))
    except Exception as e:
        record("batched_decode_correctness", False, f"ERROR {e!r}")

def t_longctx(base, model):
    # needle-in-haystack near the context window: plant a fact in a long filler prompt, ask to recall.
    try:
        # Sized to fit under --max-model-len 4096 (~3.0K prompt tok + 150 output).
        secret = "The launch code is TANGERINE-9173."
        lines = []
        for i in range(200):
            lines.append(f"Note {i}: routine status nominal; no action required for subsystem {i%17}.")
            if i == 100:
                lines.append(secret)
        prompt = "Read the notes, then answer the question.\n" + "\n".join(lines) + \
                 "\n\nQuestion: What is the launch code? Answer with just the code."
        d = chat(base, model, [{"role":"user","content":prompt}], max_tokens=150, timeout=600)
        out = d["choices"][0]["message"]["content"]
        ptoks = d["usage"]["prompt_tokens"]
        record("long_context_recall", "TANGERINE-9173" in out, f"prompt_tokens={ptoks} out={out[:80]!r}")
    except Exception as e:
        record("long_context_recall", False, f"ERROR {e!r}")

def t_determinism(base, model):
    try:
        a = chat(base, model, [{"role":"user","content":"Name three primary colors."}], max_tokens=64, timeout=300)["choices"][0]["message"]["content"]
        b = chat(base, model, [{"role":"user","content":"Name three primary colors."}], max_tokens=64, timeout=300)["choices"][0]["message"]["content"]
        record("temp0_determinism", a == b, f"identical={a==b}")
    except Exception as e:
        record("temp0_determinism", False, f"ERROR {e!r}")

def t_stop_seq(base, model):
    try:
        p = {"model":model,"messages":[{"role":"user","content":"Say: alpha beta gamma delta"}],
             "temperature":0,"max_tokens":64,"stop":["gamma"]}
        d = post(base, "/chat/completions", p, timeout=300)
        out = d["choices"][0]["message"]["content"]; fin = d["choices"][0]["finish_reason"]
        record("stop_sequence", "gamma" not in out and "delta" not in out, f"finish={fin} out={out!r}")
    except Exception as e:
        record("stop_sequence", False, f"ERROR {e!r}")

def t_reasoning(base, model):
    # informational: does raw output contain <think> tags? (affects whether OpenWebUI needs a reasoning parser)
    try:
        d = chat(base, model, [{"role":"user","content":"What is 17 plus 25?"}], max_tokens=512, timeout=400)
        out = d["choices"][0]["message"]["content"]
        has_think = "<think>" in out or "</think>" in out
        rc = d["choices"][0]["message"].get("reasoning_content")
        record("reasoning_tokens", None, f"has_<think>_tags={has_think} reasoning_content_field={'yes' if rc else 'no'} (informational)")
    except Exception as e:
        record("reasoning_tokens", None, f"ERROR {e!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    print(f"=== feature_test against {base} model={a.model} label={a.label} ===", flush=True)
    t0 = time.time()
    for fn in (t_basic, t_eos, t_multiturn, t_streaming, t_length, t_batched, t_longctx, t_determinism, t_stop_seq, t_reasoning):
        fn(base, a.model)
    dur = time.time()-t0
    # markdown
    md = [f"## Feature test — {a.label or a.model}",
          f"_endpoint: {base} · {time.strftime('%Y-%m-%d %H:%M')} · {dur:.0f}s_", "",
          "| feature | result | note |", "|---|---|---|"]
    for name, passed, note in RESULTS:
        r = "✅ PASS" if passed else ("ℹ️ INFO" if passed is None else "❌ FAIL")
        md.append(f"| {name} | {r} | {note} |")
    md = "\n".join(md)+"\n"
    print("\n"+md)
    if a.out:
        with open(a.out,"w") as f: f.write(md)
        print(f"saved: {a.out}")
    nfail = sum(1 for _,p,_ in RESULTS if p is False)
    print(f"=== {len(RESULTS)} tests, {nfail} FAIL ===")

if __name__ == "__main__":
    main()
