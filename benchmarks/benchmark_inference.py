import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_model_params
from inference.engine import MiniMindInferenceEngine

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling
    ))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def benchmark_baseline(args):
    model, tokenizer = init_model(args)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    prompt_len = len(inputs['input_ids'][0])
    
    total_latencies = []
    max_gpu_memory = 0
    all_generated_tokens = []
    
    for i in range(args.repeat):
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            use_cache=True
        )
        
        if args.device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        generated_tokens = len(generated_ids[0]) - prompt_len
        tokens_per_second = generated_tokens / (end_time - start_time)
        
        if args.device == 'cuda':
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            max_gpu_memory = max(max_gpu_memory, gpu_memory)
        
        total_latencies.append(latency_ms)
        all_generated_tokens.append(generated_tokens)
    
    avg_latency = sum(total_latencies) / len(total_latencies)
    avg_tokens = sum(all_generated_tokens) / len(all_generated_tokens)
    avg_tokens_per_second = avg_tokens / (avg_latency / 1000)
    
    return {
        'mode': 'baseline',
        'prompt_len': prompt_len,
        'max_new_tokens': args.max_new_tokens,
        'ttft_ms': 'N/A',
        'prefill_ms': 'N/A',
        'decode_ms': 'N/A',
        'total_latency_ms': avg_latency,
        'total_tokens_per_second': avg_tokens_per_second,
        'decode_tokens_per_second': 'N/A',
        'generated_tokens': int(avg_tokens),
        'max_gpu_memory_mb': max_gpu_memory if args.device == 'cuda' else 'N/A'
    }

def benchmark_engine_fast(args):
    engine = MiniMindInferenceEngine(
        weight=args.weight,
        load_from=args.load_from,
        save_dir=args.save_dir,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling,
        device=args.device
    )
    
    encoded = engine.tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    prompt_len = input_ids.shape[1]
    
    total_latencies = []
    max_gpu_memory = 0
    all_generated_tokens = []
    
    for i in range(args.repeat):
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        text = engine.generate_full_fast(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        if args.device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        generated_tokens = len(engine.tokenizer.encode(text))
        
        if args.device == 'cuda':
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            max_gpu_memory = max(max_gpu_memory, gpu_memory)
        
        total_latencies.append(latency_ms)
        all_generated_tokens.append(generated_tokens)
    
    avg_latency = sum(total_latencies) / len(total_latencies)
    avg_tokens = sum(all_generated_tokens) / len(all_generated_tokens)
    avg_tokens_per_second = avg_tokens / (avg_latency / 1000)
    
    return {
        'mode': 'engine_fast',
        'prompt_len': prompt_len,
        'max_new_tokens': args.max_new_tokens,
        'ttft_ms': 'N/A',
        'prefill_ms': 'N/A',
        'decode_ms': 'N/A',
        'total_latency_ms': avg_latency,
        'total_tokens_per_second': avg_tokens_per_second,
        'decode_tokens_per_second': 'N/A',
        'generated_tokens': int(avg_tokens),
        'max_gpu_memory_mb': max_gpu_memory if args.device == 'cuda' else 'N/A'
    }

def benchmark_engine(args):
    engine = MiniMindInferenceEngine(
        weight=args.weight,
        load_from=args.load_from,
        save_dir=args.save_dir,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling,
        device=args.device
    )
    
    encoded = engine.tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    prompt_len = input_ids.shape[1]
    
    total_latencies = []
    total_ttft = []
    total_prefill_times = []
    total_decode_times = []
    total_decode_tokens = []
    max_gpu_memory = 0
    all_generated_tokens = []
    
    for i in range(args.repeat):
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        total_start = time.time()
        
        prefill_start = time.time()
        logits, kv_cache = engine.prefill(input_ids)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        prefill_end = time.time()
        
        next_token_logits = logits[:, -1, :] / args.temperature
        next_token = engine._sample(next_token_logits, args.top_p)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        first_token_end = time.time()
        
        prefill_ms = (prefill_end - prefill_start) * 1000
        ttft_ms = (first_token_end - prefill_start) * 1000
        
        generated_tokens = 1
        
        decode_start = time.time()
        decode_tokens_count = 0
        for _ in range(args.max_new_tokens - 1):
            logits, kv_cache = engine.decode_one(next_token, kv_cache)
            next_token_logits = logits[:, -1, :] / args.temperature
            next_token = engine._sample(next_token_logits, args.top_p)
            generated_tokens += 1
            decode_tokens_count += 1
            
            if next_token[0] == engine.tokenizer.eos_token_id:
                break
        
        if args.device == 'cuda':
            torch.cuda.synchronize()
        
        total_end = time.time()
        
        total_latency_ms = (total_end - total_start) * 1000
        decode_ms = (total_end - decode_start) * 1000
        
        if args.device == 'cuda':
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            max_gpu_memory = max(max_gpu_memory, gpu_memory)
        
        total_latencies.append(total_latency_ms)
        total_ttft.append(ttft_ms)
        total_prefill_times.append(prefill_ms)
        total_decode_times.append(decode_ms)
        total_decode_tokens.append(decode_tokens_count)
        all_generated_tokens.append(generated_tokens)
    
    avg_latency = sum(total_latencies) / len(total_latencies)
    avg_ttft = sum(total_ttft) / len(total_ttft)
    avg_prefill_ms = sum(total_prefill_times) / len(total_prefill_times)
    avg_decode_ms = sum(total_decode_times) / len(total_decode_times)
    avg_tokens = sum(all_generated_tokens) / len(all_generated_tokens)
    avg_total_tokens_per_second = avg_tokens / (avg_latency / 1000)
    
    avg_decode_tokens = sum(total_decode_tokens) / len(total_decode_tokens)
    avg_decode_tokens_per_second = avg_decode_tokens / (avg_decode_ms / 1000) if avg_decode_ms > 0 else 0.0
    
    return {
        'mode': 'engine',
        'prompt_len': prompt_len,
        'max_new_tokens': args.max_new_tokens,
        'ttft_ms': avg_ttft,
        'prefill_ms': avg_prefill_ms,
        'decode_ms': avg_decode_ms,
        'total_latency_ms': avg_latency,
        'total_tokens_per_second': avg_total_tokens_per_second,
        'decode_tokens_per_second': avg_decode_tokens_per_second,
        'generated_tokens': int(avg_tokens),
        'max_gpu_memory_mb': max_gpu_memory if args.device == 'cuda' else 'N/A'
    }

def main():
    parser = argparse.ArgumentParser(description="MiniMind推理基准测试")
    parser.add_argument('--load_from', default='model', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true')
    parser.add_argument('--prompt', default='你好，请介绍一下你自己', type=str)
    parser.add_argument('--max_new_tokens', default=128, type=int)
    parser.add_argument('--repeat', default=5, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--mode', default='baseline', type=str, choices=['baseline', 'engine', 'engine_fast'])
    args = parser.parse_args()
    
    print(f"=== 推理基准测试 [{args.mode}] ===")
    print(f"Prompt: {args.prompt}")
    print(f"重复次数: {args.repeat}")
    print(f"设备: {args.device}")
    print("=" * 80)
    
    if args.mode == 'baseline':
        result = benchmark_baseline(args)
    elif args.mode == 'engine_fast':
        result = benchmark_engine_fast(args)
    else:
        result = benchmark_engine(args)
    
    print(f"| mode | prompt_len | max_new_tokens | TTFT(ms) | prefill(ms) | decode(ms) | total(ms) | total tok/s | decode tok/s | memory(MB) |")
    print(f"|------|-----------:|---------------:|----------:|------------:|-----------:|----------:|------------:|-------------:|-----------:|")
    
    ttft_str = f"{result['ttft_ms']:.2f}" if result['ttft_ms'] != 'N/A' else 'N/A'
    prefill_str = f"{result['prefill_ms']:.2f}" if result['prefill_ms'] != 'N/A' else 'N/A'
    decode_str = f"{result['decode_ms']:.2f}" if result['decode_ms'] != 'N/A' else 'N/A'
    total_tok_s_str = f"{result['total_tokens_per_second']:.2f}" if result['total_tokens_per_second'] != 'N/A' else 'N/A'
    decode_tok_s_str = f"{result['decode_tokens_per_second']:.2f}" if result['decode_tokens_per_second'] != 'N/A' else 'N/A'
    memory_str = f"{result['max_gpu_memory_mb']:.2f}" if result['max_gpu_memory_mb'] != 'N/A' else 'N/A'
    
    print(f"| {result['mode']} | {result['prompt_len']} | {result['max_new_tokens']} | {ttft_str} | {prefill_str} | {decode_str} | {result['total_latency_ms']:.2f} | {total_tok_s_str} | {decode_tok_s_str} | {memory_str} |")

if __name__ == "__main__":
    main()