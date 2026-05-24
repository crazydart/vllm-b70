## Feature test — gemma-4-E4B v0.21 + transformers v5 TP=2 eager
_endpoint: http://127.0.0.1:8080/v1 · 2026-05-24 23:03 · 153s_

| feature | result | note |
|---|---|---|
| basic_generation | ✅ PASS | finish=stop out_tail='PONG' |
| natural_eos | ✅ PASS | finish=stop (want 'stop', not 'length'), completion_tokens=8 |
| multi_turn_context | ✅ PASS | out='94' |
| streaming_sse | ✅ PASS | chunks=16 done=True len=14 |
| correctness_at_length | ✅ PASS | words=157 uniq_ratio=0.67 (want >0.30) |
| batched_decode_correctness | ✅ PASS | 4/4 correct: tokyo:Y | cairo:Y | rome:Y | 42:Y |
| long_context_recall | ✅ PASS | prompt_tokens=3817 out='TANGERINE-9173' |
| temp0_determinism | ✅ PASS | identical=True |
| stop_sequence | ✅ PASS | finish=stop out='Alpha beta ' |
| reasoning_tokens | ℹ️ INFO | has_<think>_tags=False reasoning_content_field=no (informational) |
