## Feature test — v0.20.2 TP=4 compiled
_endpoint: http://127.0.0.1:8080/v1 · 2026-05-24 17:50 · 373s_

| feature | result | note |
|---|---|---|
| basic_generation | ✅ PASS | finish=stop out_tail='.\n   - Any extra characters? No.\n   - Ready.✅\n</think>\n\nPONG' |
| natural_eos | ✅ PASS | finish=stop (want 'stop', not 'length'), completion_tokens=112 |
| multi_turn_context | ✅ PASS | out='Here\'s a thinking process:\n\n1.  **User Input:** "What is my favorite number times 2? Answer with just the number."\n2.  *' |
| streaming_sse | ✅ PASS | chunks=206 done=True len=642 |
| correctness_at_length | ❌ FAIL | words=1 uniq_ratio=1.00 (want >0.30) |
| batched_decode_correctness | ❌ FAIL | 0/4 correct: tokyo:N | cairo:N | rome:N | 42:N |
| long_context_recall | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| temp0_determinism | ✅ PASS | identical=True |
| stop_sequence | ✅ PASS | finish=length out='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' |
| reasoning_tokens | ℹ️ INFO | has_<think>_tags=False reasoning_content_field=no (informational) |
