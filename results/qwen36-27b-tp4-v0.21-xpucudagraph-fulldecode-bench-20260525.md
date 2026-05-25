[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (159385 > 1024). Running this sequence through the model will result in indexing errors
llama-benchy (0.3.7)
Date: 2026-05-25 16:31:14
Benchmarking model: qwen-xpugraph at http://127.0.0.1:8080/v1
Concurrency levels: [1, 2, 4]
Error loading tokenizer: qwen-xpugraph is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
Falling back to 'gpt2' tokenizer as approximation.
Loading text from cache: /home/player1/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt
Total tokens available in text corpus: 159385
Warming up...
Warmup (User only) complete. Delta: 8 tokens (Server: 30, Local: 22)
Warmup (System+Empty) complete. Delta: 13 tokens (Server: 35, Local: 22)

Running coherence test...
Coherence test PASSED.
Measuring latency using mode: api...
Average latency (api): 3.21 ms
Running test: pp=128, tg=256, depth=0, concurrency=1
  Run 1/3 (batch size 1)...
  Run 2/3 (batch size 1)...
  Run 3/3 (batch size 1)...
Running test: pp=128, tg=256, depth=0, concurrency=2
  Run 1/3 (batch size 2)...
  Run 2/3 (batch size 2)...
  Run 3/3 (batch size 2)...
Running test: pp=128, tg=256, depth=0, concurrency=4
  Run 1/3 (batch size 4)...
  Run 2/3 (batch size 4)...
  Run 3/3 (batch size 4)...
Printing results in MD format:



| model         |       test |     t/s (total) |       t/s (req) |      peak t/s |   peak t/s (req) |         ttfr (ms) |      est_ppt (ms) |     e2e_ttft (ms) |
|:--------------|-----------:|----------------:|----------------:|--------------:|-----------------:|------------------:|------------------:|------------------:|
| qwen-xpugraph | pp128 (c1) |  520.52 ± 43.57 |  520.52 ± 43.57 |               |                  |    233.35 ± 11.05 |    230.14 ± 11.05 |    233.35 ± 11.05 |
| qwen-xpugraph | tg256 (c1) |    26.06 ± 0.27 |    26.06 ± 0.27 |  27.67 ± 0.47 |     27.67 ± 0.47 |                   |                   |                   |
| qwen-xpugraph | pp128 (c2) | 203.93 ± 166.80 | 132.68 ± 136.45 |               |                  | 3890.95 ± 4191.50 | 3887.74 ± 4191.50 | 3890.95 ± 4191.50 |
| qwen-xpugraph | tg256 (c2) |    50.16 ± 0.86 |    25.23 ± 0.24 |  54.00 ± 0.00 |     27.00 ± 0.00 |                   |                   |                   |
| qwen-xpugraph | pp128 (c4) |  203.65 ± 47.58 | 140.33 ± 149.78 |               |                  |  1762.00 ± 961.26 |  1758.79 ± 961.26 |  1762.00 ± 961.26 |
| qwen-xpugraph | tg256 (c4) |   77.88 ± 11.57 |    23.04 ± 1.54 | 100.00 ± 8.64 |     26.28 ± 0.53 |                   |                   |                   |

llama-benchy (0.3.7)
date: 2026-05-25 16:31:14 | latency mode: api
