# MiniMind 小模型训练与推理工程化改造

本项目基于 MiniMind，完成从训练复现到推理优化、Benchmark 评测、CLI/API 服务化的一整套工程化改造。

## 项目定位

将 MiniMind 从一个训练/推理演示仓库，升级为一个可评测、可优化、可服务化的小模型实验平台。

核心改造包括：

- RTX 5090 上完成 Pretrain + Full SFT
- 显式 KV Cache 封装
- Prefill / Decode 推理阶段拆分
- MiniMindInferenceEngine 统一推理引擎
- Benchmark 对比 baseline / engine / engine_fast
- CLI 推理入口统一走 engine
- OpenAI 风格 `/v1/chat/completions` API 服务
- 支持 SSE 流式输出与 `data: [DONE]`

## 训练结果

| 阶段 | 数据 | 产物 |
|---|---|---|
| Pretrain | `pretrain_t2t_mini.jsonl` | `out/pretrain_768.pth` |
| Full SFT | `sft_t2t_mini.jsonl` | `out/full_sft_768.pth` |

模型参数量约为 `63.91M`。

## 推理架构

推理统一封装在 `MiniMindInferenceEngine` 中：

```text
prompt
  -> prefill(input_ids)
  -> KV Cache
  -> decode_one(new_token, kv_cache)
  -> generate_stream / generate_full_fast
```

当前统一入口：

```text
benchmarks/benchmark_inference.py  -> MiniMindInferenceEngine
eval_llm.py                        -> MiniMindInferenceEngine
scripts/serve_openai_api.py        -> MiniMindInferenceEngine
```

## Benchmark 结果

测试环境：

- GPU: NVIDIA GeForce RTX 5090
- 权重: full_sft
- prompt_len: 7
- max_new_tokens: 128
- repeat: 5
- device: cuda

| mode | total(ms) | total tok/s | memory(MB) |
|---|---:|---:|---:|
| baseline | 908.84 | 140.84 | 146.05 |
| engine | 1289.74 | 86.53 | 382.09 |
| engine_fast | 677.71 | 151.10 | 146.06 |

结论：

- `engine_fast` 相比 baseline 提升约 `7.3%`
- `engine_fast` 相比初版 engine 提升约 `74.7%`
- 贪心解码相比采样版只提升约 `5%`，说明当前主要瓶颈不是 top-p 采样

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
pip install fastapi uvicorn pydantic streamlit -i https://mirrors.aliyun.com/pypi/simple
```

### 2. 数据下载

```bash
mkdir -p dataset

aria2c -x 16 -s 16 -k 1M --continue=true --file-allocation=none \
"https://hf-mirror.com/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_t2t_mini.jsonl" \
-d ./dataset -o pretrain_t2t_mini.jsonl

aria2c -x 16 -s 16 -k 1M --continue=true --file-allocation=none \
"https://hf-mirror.com/datasets/jingyaogong/minimind_dataset/resolve/main/sft_t2t_mini.jsonl" \
-d ./dataset -o sft_t2t_mini.jsonl
```

### 3. 训练

```bash
cd trainer
nohup python train_pretrain.py > pretrain.log 2>&1 &
tail -f pretrain.log

nohup python train_full_sft.py > sft.log 2>&1 &
tail -f sft.log
```

### 4. Benchmark

```bash
python benchmarks/benchmark_inference.py --weight full_sft --mode baseline \
  --prompt "你好，请介绍一下你自己" --max_new_tokens 128 --repeat 5 --device cuda

python benchmarks/benchmark_inference.py --weight full_sft --mode engine_fast \
  --prompt "你好，请介绍一下你自己" --max_new_tokens 128 --repeat 5 --device cuda
```

### 5. CLI 推理

```bash
python eval_llm.py --weight full_sft --mode stream --max_new_tokens 128
python eval_llm.py --weight full_sft --mode fast --max_new_tokens 128
```

### 6. OpenAI API 服务

```bash
python scripts/serve_openai_api.py --weight full_sft --device cuda
```

非流式调用：

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
    "stream": false,
    "max_tokens": 128
  }'
```

流式调用：

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}],
    "stream": true,
    "max_tokens": 128
  }'
```

## 项目亮点

1. 完成 MiniMind 训练与推理全链路复现
2. 显式 KV Cache 设计，便于理解和扩展 LLM 增量推理
3. Prefill / Decode 阶段拆分，符合真实推理系统结构
4. Benchmark 结果可量化，避免只做主观优化
5. CLI、Benchmark、API 服务统一复用同一套推理引擎
6. 支持 OpenAI 风格 Chat Completions 接口和 SSE 流式输出

## 后续计划

- WebUI 接入 MiniMindInferenceEngine
- Benchmark 增加 P50/P95、不同 prompt 长度和显存曲线
- 进一步优化 attention 内部 KV Cache 拼接
- 补充 docs/training_pipeline.md 和 docs/interview_notes.md
