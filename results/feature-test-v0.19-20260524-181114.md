## Feature test — v0.19 TP=4 eager (patched)
_endpoint: http://127.0.0.1:8080/v1 · 2026-05-24 18:27 · 990s_

| feature | result | note |
|---|---|---|
| basic_generation | ❌ FAIL | finish=length out_tail='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' |
| natural_eos | ❌ FAIL | finish=length (want 'stop', not 'length'), completion_tokens=512 |
| multi_turn_context | ✅ PASS | out='Here\'s a thinking process:\n\n1.  **Analyze User Input:**\n   - User says: "What is my favorite number times 2? Answer with' |
| streaming_sse | ✅ PASS | chunks=257 done=True len=568 |
| correctness_at_length | ❌ FAIL | words=1 uniq_ratio=1.00 (want >0.30) |
| batched_decode_correctness | ❌ FAIL | 1/4 correct: tokyo:Y | cairo:N | rome:N | 42:N |
| long_context_recall | ❌ FAIL | prompt_tokens=3818 out='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' |
| temp0_determinism | ✅ PASS | identical=True |
| stop_sequence | ✅ PASS | finish=length out='Here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' |
| reasoning_tokens | ℹ️ INFO | has_<think>_tags=False reasoning_content_field=no (informational) |
