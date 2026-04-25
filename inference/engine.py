import argparse
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.kv_cache import MiniMindKVCache
from trainer.trainer_utils import get_model_params

class MiniMindInferenceEngine:
    def __init__(self, weight='full_sft', load_from='model', save_dir='out', 
                 hidden_size=768, num_hidden_layers=8, use_moe=False, 
                 inference_rope_scaling=False, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = weight
        self.load_from = load_from
        self.save_dir = save_dir
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.inference_rope_scaling = inference_rope_scaling
        
        self.tokenizer = None
        self.model = None
        self._init_model()
    
    def _init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.load_from)
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            use_moe=self.use_moe,
            inference_rope_scaling=self.inference_rope_scaling
        ))
        moe_suffix = '_moe' if self.use_moe else ''
        ckp = f'./{self.save_dir}/{self.weight}_{self.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=self.device), strict=True)
        get_model_params(model, model.config)
        self.model = model.half().eval().to(self.device)
    
    def generate_full(self, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            top_p=top_p,
            temperature=temperature,
            use_cache=True
        )
        response = self.tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response
    
    def prefill(self, input_ids):
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=True
        )
        kv_cache = MiniMindKVCache.from_past_key_values(outputs.past_key_values)
        return outputs.logits, kv_cache
    
    def decode_one(self, input_id, kv_cache):
        past_key_values = kv_cache.to_past_key_values()

        outputs = self.model.forward(
            input_ids=input_id,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True
        )

        kv_cache.update_from_past_key_values_(outputs.past_key_values)

        return outputs.logits, kv_cache
    
    @torch.inference_mode()
    def generate_stream(
        self,
        prompt,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        decode_interval=4,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        logits, kv_cache = self.prefill(input_ids)

        next_token_logits = logits[:, -1, :] / temperature
        next_token = self._sample(next_token_logits, top_p)

        eos_token_id = self.tokenizer.eos_token_id
        buffer_tokens = []

        for _ in range(max_new_tokens):
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            buffer_tokens.append(next_token)

            if len(buffer_tokens) >= decode_interval:
                chunk_ids = torch.cat(buffer_tokens, dim=-1)
                text = self.tokenizer.decode(chunk_ids[0], skip_special_tokens=True)
                if text:
                    yield text
                buffer_tokens.clear()

            logits, kv_cache = self.decode_one(next_token, kv_cache)

            next_token_logits = logits[:, -1, :] / temperature
            next_token = self._sample(next_token_logits, top_p)

        if buffer_tokens:
            chunk_ids = torch.cat(buffer_tokens, dim=-1)
            text = self.tokenizer.decode(chunk_ids[0], skip_special_tokens=True)
            if text:
                yield text
    
    @torch.inference_mode()
    def generate_full_fast(
        self,
        prompt,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        logits, kv_cache = self.prefill(input_ids)

        next_token_logits = logits[:, -1, :] / temperature
        next_token = self._sample(next_token_logits, top_p)

        output_tokens = []

        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            output_tokens.append(next_token)

            logits, kv_cache = self.decode_one(next_token, kv_cache)

            next_token_logits = logits[:, -1, :] / temperature
            next_token = self._sample(next_token_logits, top_p)

        if not output_tokens:
            return ""

        output_ids = torch.cat(output_tokens, dim=-1)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _sample(self, logits, top_p):
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0
            logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def generate_full_greedy_fast(
        self,
        prompt,
        max_new_tokens=128,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        logits, kv_cache = self.prefill(input_ids)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        output_tokens = []
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            output_tokens.append(next_token)

            logits, kv_cache = self.decode_one(next_token, kv_cache)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        if not output_tokens:
            return ""

        output_ids = torch.cat(output_tokens, dim=-1)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="MiniMind推理引擎测试")
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--load_from', default='model', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true')
    parser.add_argument('--prompt', default='你好', type=str)
    parser.add_argument('--max_new_tokens', default=128, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--stream', action='store_true', help='启用流式输出')
    args = parser.parse_args()
    
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
    
    print(f"Prompt: {args.prompt}")
    print("Response: ", end='', flush=True)
    
    if args.stream:
        full_response = ""
        for token_text in engine.generate_stream(args.prompt, max_new_tokens=args.max_new_tokens,
                                                temperature=args.temperature, top_p=args.top_p):
            print(token_text, end='', flush=True)
            full_response += token_text
        print()
    else:
        response = engine.generate_full(args.prompt, max_new_tokens=args.max_new_tokens, 
                                       temperature=args.temperature, top_p=args.top_p)
        print(response)

if __name__ == "__main__":
    main()