## Feature test — Qwen3.6-35B-A3B MoE v0.21 TP=4 eager
_endpoint: http://127.0.0.1:8080/v1 · 2026-05-24 22:22 · 522s_

| feature | result | note |
|---|---|---|
| basic_generation | ✅ PASS | finish=stop out_tail='nation.\n5.  **Final Output Generation:** PONG\n</think>\n\nPONG' |
| natural_eos | ✅ PASS | finish=stop (want 'stop', not 'length'), completion_tokens=155 |
| multi_turn_context | ✅ PASS | out='Here\'s a thinking process:\n\n1.  **Analyze User Input:**\n   - User previously stated: "My favorite number is 47. Remember' |
| streaming_sse | ✅ PASS | chunks=217 done=True len=671 |
| correctness_at_length | ✅ PASS | words=188 uniq_ratio=0.73 (want >0.30) |
| batched_decode_correctness | ✅ PASS | 4/4 correct: tokyo:Y | cairo:Y | rome:Y | 42:Y |
| long_context_recall | ✅ PASS | prompt_tokens=3818 out='The user wants to find the launch code from the provided notes.\nI will scan the ' |
| temp0_determinism | ✅ PASS | identical=True |
| stop_sequence | ✅ PASS | finish=stop out='Here\'s a thinking process:\n\n1.  **Analyze User Input:**\n   - User says: "Say: alpha beta ' |
| reasoning_tokens | ℹ️ INFO | has_<think>_tags=True reasoning_content_field=no (informational) |
