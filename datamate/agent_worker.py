"""Generation worker: python -m datamate.agent_worker --model PATH --in jobs.jsonl --out out.jsonl [--tp 2] [--thinking]
Loads one vLLM model, answers every job (batched), writes {"id", "texts"} lines, exits (releasing the GPUs)."""
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2); ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--quantization", default=None, help="fp8 (on-the-fly, native on Blackwell) | awq | gptq | None (pre-quantized checkpoints are auto-detected)")
    ap.add_argument("--dtype", default="auto", help="auto | float16 (T4/V100: no bf16) | bfloat16")
    ap.add_argument("--enforce-eager", action="store_true", help="skip CUDA-graph capture (saves memory/startup on small GPUs)")
    ap.add_argument("--no-prefix-cache", action="store_true", help="disable prefix caching (fallback for old GPUs / backends)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    jobs = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    print(f"[worker] {len(jobs)} jobs, model={a.model}, max_len={a.max_model_len}, dtype={a.dtype}, "
          f"eager={a.enforce_eager}, prefix_cache={not a.no_prefix_cache}", flush=True)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, seed=0, gpu_memory_utilization=a.gpu_util,
              quantization=a.quantization, enable_prefix_caching=not a.no_prefix_cache,
              dtype=a.dtype, enforce_eager=a.enforce_eager)   # prefix caching: rulebook/schema prefix shared by K samples
    sps = [SamplingParams(n=j.get("n", 1), temperature=j.get("temperature", 0.0), top_p=0.95 if j.get("temperature", 0) > 0 else 1.0,
                          max_tokens=j.get("max_tokens", 1500), seed=j.get("seed", 0)) for j in jobs]
    # keep every prompt inside the context window: drop text just before the question (end of the schema section)
    tok = llm.get_tokenizer()
    for j, sp in zip(jobs, sps):
        budget = a.max_model_len - sp.max_tokens - 64
        msgs = j["messages"]
        for _ in range(30):
            n = len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
            if n <= budget:
                break
            m0 = msgs[0]["content"]
            cut = int(len(m0) * (n - budget) / n) + 200
            i = m0.find("## Nhiệm vụ")
            if i < 0:
                i = len(m0) // 2
            msgs[0]["content"] = m0[: max(0, i - cut)] + "\n...(bớt bảng mẫu cho vừa ngữ cảnh)...\n" + m0[i:]
    kw = {"chat_template_kwargs": {"enable_thinking": a.thinking}}
    print("[worker] model loaded, generating", flush=True)
    outs = llm.chat([j["messages"] for j in jobs], sps, **kw)
    with open(a.out, "w", encoding="utf-8") as f:
        for j, o in zip(jobs, outs):
            f.write(json.dumps({"id": j["id"], "texts": [c.text for c in o.outputs]}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
