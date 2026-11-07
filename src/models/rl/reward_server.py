import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", "Qwen/Qwen3-8B")

app = FastAPI()
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from vllm import LLM
        _llm = LLM(
            model=JUDGE_MODEL_NAME,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
        )
    return _llm


def extract_bool_from_text(text: str) -> int:
    t = text.strip().lower()
    if "false" in t:
        return 0
    if "true" in t:
        return 1
    return 0


@app.post("/judge")
async def judge(request: Request):
    data = await request.json()
    prompts = data.get("prompts", [])

    if not prompts:
        return JSONResponse({"scores": []})

    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=5)
    llm = get_llm()
    tokenizer = llm.get_tokenizer()
    # Format each instruction as a Qwen chat turn and disable thinking for verdicts.
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(chat_prompts, sampling_params)

    scores = []
    for out in outputs:
        text = out.outputs[0].text
        scores.append(float(extract_bool_from_text(text)))

    return JSONResponse({"scores": scores})
