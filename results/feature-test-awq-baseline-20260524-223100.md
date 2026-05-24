## Feature test — Qwen3.6-27B AWQ-INT4 v0.21 TP=2 eager (no MTP)
_endpoint: http://127.0.0.1:8080/v1 · 2026-05-24 22:39 · 524s_

| feature | result | note |
|---|---|---|
| basic_generation | ✅ PASS | finish=stop out_tail='exactly? Yes.\n5.  **Output Generation:** PONG\n</think>\n\nPONG' |
| natural_eos | ✅ PASS | finish=stop (want 'stop', not 'length'), completion_tokens=155 |
| multi_turn_context | ✅ PASS | out='Here\'s a thinking process:\n\n1.  **Analyze User Input:**\n   - User says: "What is my favorite number times 2? Answer with' |
| streaming_sse | ✅ PASS | chunks=199 done=True len=541 |
| correctness_at_length | ✅ PASS | words=184 uniq_ratio=0.77 (want >0.30) |
| batched_decode_correctness | ✅ PASS | 4/4 correct: tokyo:Y | cairo:Y | rome:Y | 42:Y |
| long_context_recall | ✅ PASS | prompt_tokens=3818 out='The user wants to find the launch code from the provided text.\nI will scan the t' |
| temp0_determinism | ✅ PASS | identical=True |
| stop_sequence | ✅ PASS | finish=stop out='Here\'s a thinking process:\n\n1.  **Analyze User Input:**\n   - User says: "Say: alpha beta ' |
| reasoning_tokens | ℹ️ INFO | has_<think>_tags=True reasoning_content_field=no (informational) |
