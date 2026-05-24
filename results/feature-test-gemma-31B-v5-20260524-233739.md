## Feature test — gemma-4-31B dense v0.21+v5 TP=4 eager
_endpoint: http://127.0.0.1:8080/v1 · 2026-05-24 23:37 · 0s_

| feature | result | note |
|---|---|---|
| basic_generation | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| natural_eos | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| multi_turn_context | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| streaming_sse | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| correctness_at_length | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| batched_decode_correctness | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| long_context_recall | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| temp0_determinism | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| stop_sequence | ❌ FAIL | ERROR <HTTPError 400: 'Bad Request'> |
| reasoning_tokens | ℹ️ INFO | ERROR <HTTPError 400: 'Bad Request'> |
