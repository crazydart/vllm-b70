## Feature test — v0.20.2 TP=4 compiled
_endpoint: http://127.0.0.1:8080/v1 · 2026-05-24 17:43 · 252s_

| feature | result | note |
|---|---|---|
| basic_generation | ❌ FAIL | finish=length out="Here's a thinking process:\n\n1.  **Analyze User Input:**" |
| natural_eos | ✅ PASS | finish=stop (want 'stop', not 'length'), completion_tokens=112 |
| multi_turn_context | ✅ PASS | out='Here\'s a thinking process:\n\n1.  **User Input:** "What is my favorite number times 2? Answer with just the number."\n2.  *' |
| streaming_sse | ✅ PASS | chunks=202 done=True len=554 |
| correctness_at_length | ✅ PASS | words=188 uniq_ratio=0.78 (want >0.30) |
| batched_decode_correctness | ✅ PASS | 4/4 correct: tokyo:Y | cairo:Y | rome:Y | 42:Y |
| long_context_recall | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| temp0_determinism | ✅ PASS | identical=True |
| stop_sequence | ✅ PASS | finish=stop out='Here\'s a thinking process:\n\n1.  **Analyze User Input:**\n   - User says: "Say: alpha beta ' |
| reasoning_tokens | ℹ️ INFO | has_<think>_tags=True reasoning_content_field=no (informational) |
