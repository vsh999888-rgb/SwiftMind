import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import torch
from inference.engine import MiniMindInferenceEngine

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "你好，请介绍一下你自己"
max_new_tokens = 128
repeat = 5

engine = MiniMindInferenceEngine(
    weight='full_sft',
    load_from='model',
    save_dir='out',
    hidden_size=768,
    num_hidden_layers=8,
    use_moe=False,
    inference_rope_scaling=False,
    device=device
)

print(f"Device: {device}")
print(f"Prompt: {prompt}")
print(f"Max new tokens: {max_new_tokens}")
print(f"Repeat: {repeat}")
print("=" * 60)

print("\n--- Testing generate_full_fast (with sampling) ---")
# warmup
_ = engine.generate_full_fast(prompt, max_new_tokens=32)
if device == 'cuda':
    torch.cuda.synchronize()

times = []
for i in range(repeat):
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    text = engine.generate_full_fast(prompt, max_new_tokens=max_new_tokens)

    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    times.append(elapsed)

    print(f"Run {i+1}: {elapsed * 1000:.2f} ms, {max_new_tokens / elapsed:.2f} tok/s")

avg = sum(times) / len(times)
print("-" * 60)
print(f"Average: {avg * 1000:.2f} ms")
print(f"Tokens/s: {max_new_tokens / avg:.2f}")
preview_text = text[:100].replace('\n', ' ')
print(f"Sample output: {preview_text}")

print("\n--- Testing generate_full_greedy_fast (no sampling) ---")
# warmup
_ = engine.generate_full_greedy_fast(prompt, max_new_tokens=32)
if device == 'cuda':
    torch.cuda.synchronize()

times = []
for i in range(repeat):
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    text = engine.generate_full_greedy_fast(prompt, max_new_tokens=max_new_tokens)

    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    times.append(elapsed)

    print(f"Run {i+1}: {elapsed * 1000:.2f} ms, {max_new_tokens / elapsed:.2f} tok/s")

avg = sum(times) / len(times)
print("-" * 60)
print(f"Average: {avg * 1000:.2f} ms")
print(f"Tokens/s: {max_new_tokens / avg:.2f}")
preview_text = text[:100].replace('\n', ' ')
print(f"Sample output: {preview_text}")