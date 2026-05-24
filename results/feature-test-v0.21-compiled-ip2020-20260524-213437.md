## Feature test — v0.21.0 TP=4 COMPILED+IP20.2.0
_endpoint: http://127.0.0.1:8080/v1 · 2026-05-24 21:40 · 365s_

| feature | result | note |
|---|---|---|
| basic_generation | ✅ PASS | finish=stop out_tail='.\n   - Any extra characters? No.\n   - Ready.✅\n</think>\n\nPONG' |
| natural_eos | ✅ PASS | finish=stop (want 'stop', not 'length'), completion_tokens=112 |
| multi_turn_context | ✅ PASS | out='Here\'s a thinking process:\n\n1.  **User Input:** "What is my favorite number times 2? Answer with just the number."\n2.  *' |
| streaming_sse | ✅ PASS | chunks=202 done=True len=554 |
| correctness_at_length | ✅ PASS | words=186 uniq_ratio=0.72 (want >0.30) |
| batched_decode_correctness | ✅ PASS | 4/4 correct: tokyo:Y | cairo:Y | rome:Y | 42:Y |
| long_context_recall | ✅ PASS | prompt_tokens=3818 out='The user wants to extract the launch code from the provided text.\nI will scan th' |
| temp0_determinism | ✅ PASS | identical=True |
| stop_sequence | ✅ PASS | finish=stop out='Here\'s a thinking process:\n\n1.  **Analyze User Input:**\n   - User says: "Say: alpha beta ' |
| reasoning_tokens | ℹ️ INFO | has_<think>_tags=True reasoning_content_field=no (informational) |
