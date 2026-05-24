## Feature test — Qwen3.6-27B AWQ-INT4 + MTP v0.21 TP=2
_endpoint: http://127.0.0.1:8080/v1 · 2026-05-24 22:50 · 39s_

| feature | result | note |
|---|---|---|
| basic_generation | ❌ FAIL | ERROR <HTTPError 500: 'Internal Server Error'> |
| natural_eos | ❌ FAIL | ERROR <HTTPError 500: 'Internal Server Error'> |
| multi_turn_context | ❌ FAIL | ERROR <HTTPError 500: 'Internal Server Error'> |
| streaming_sse | ❌ FAIL | ERROR <HTTPError 500: 'Internal Server Error'> |
| correctness_at_length | ❌ FAIL | ERROR <HTTPError 500: 'Internal Server Error'> |
| batched_decode_correctness | ❌ FAIL | ERROR <HTTPError 500: 'Internal Server Error'> |
| long_context_recall | ❌ FAIL | ERROR <HTTPError 500: 'Internal Server Error'> |
| temp0_determinism | ❌ FAIL | ERROR <HTTPError 500: 'Internal Server Error'> |
| stop_sequence | ❌ FAIL | ERROR <HTTPError 500: 'Internal Server Error'> |
| reasoning_tokens | ℹ️ INFO | ERROR <HTTPError 500: 'Internal Server Error'> |
