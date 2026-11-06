"""Side lab: Gemini qua OpenAI-compatible API; giữ hợp đồng tên hàm của đề.

Không gọi API khi import. Thêm GEMINI_API_KEY trong môi trường hoặc Colab Secrets.
Các khoá gpt4o_* / tiêu đề GPT-4o được giữ để tương thích test, mang nghĩa model lớn.
"""

import os
import time
from typing import Any, Callable

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
if LLM_PROVIDER not in {"gemini", "openai"}:
    raise ValueError("Side lab hỗ trợ LLM_PROVIDER=gemini hoặc openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gemini-3.5-flash" if LLM_PROVIDER == "gemini" else "gpt-6.1-terra")
OPENAI_MINI_MODEL = os.getenv("OPENAI_MINI_MODEL", "gemini-3.5-flash-lite" if LLM_PROVIDER == "gemini" else "gpt-6-luna")
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005, "gpt-6-luna": 0.0005,
    "gemini-3.8-flash": 0.00375, "gemini-3.5-flash": 0.009, "gemini-3.5-flash-lite": 0.0025,
}


def _new_client():
    from openai import OpenAI
    from pathlib import Path
    for p in [Path(".env"), Path("../.env"), Path("../../.env"), Path("../../../.env")]:
        if p.exists():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    if v.strip():
                        os.environ.setdefault(k.strip(), v.strip())
            break

    key_name = "GEMINI_API_KEY" if LLM_PROVIDER == "gemini" else "OPENAI_API_KEY"
    key = os.getenv(key_name)
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get(key_name)
        except Exception:
            key = None
    if LLM_PROVIDER == "gemini":
        # Chuỗi rỗng tránh vô tình dùng OPENAI_API_KEY của một dịch vụ khác.
        return OpenAI(api_key=key or "", base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                      timeout=30.0, max_retries=0)
    return OpenAI(api_key=key, timeout=30.0, max_retries=0)


def _generation_kwargs(max_tokens: int, model: str | None = None) -> dict:
    if LLM_PROVIDER == "gemini":
        effort = "low" if (model or OPENAI_MODEL) == "gemini-3.8-flash" else "minimal"
        return {"max_tokens": max_tokens, "reasoning_effort": effort}
    return {"max_completion_tokens": max_tokens, "reasoning_effort": "none"}


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """Đo thời gian request; nội dung rỗng được trả về chuỗi rỗng."""
    client = _new_client()
    sampling = {"temperature": temperature, "top_p": top_p}
    if LLM_PROVIDER == "gemini" and model == "gemini-3.8-flash":
        sampling = {}
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **sampling,
        **_generation_kwargs(max_tokens, model),
    )
    latency = float(time.perf_counter() - started)
    text = response.choices[0].message.content if response.choices else ""
    return text or "", latency


def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    return call_openai(prompt, model=OPENAI_MINI_MODEL, temperature=temperature,
                       top_p=top_p, max_tokens=max_tokens)


def compare_models(prompt: str) -> dict:
    large_text, large_latency = call_openai(prompt)
    small_text, small_latency = call_openai_mini(prompt)
    # Theo hợp đồng template: chỉ ước tính token OUTPUT, chưa tính input/thinking.
    output_tokens = len(large_text.split()) / 0.75
    output_rate = COST_PER_1K_OUTPUT_TOKENS.get(OPENAI_MODEL, 0.0)
    return {
        "gpt4o_response": large_text, "mini_response": small_text,
        "gpt4o_latency": large_latency, "mini_latency": small_latency,
        "gpt4o_cost_estimate": output_tokens / 1000 * output_rate,
    }


def streaming_chatbot() -> None:
    """Stream từng mảnh text; giữ 3 lượt hoàn chỉnh, thoát bằng quit/exit."""
    history = []
    client = None
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message.lower() in {"quit", "exit"}:
            return
        if not message:
            continue
        if client is None:
            client = _new_client()
        messages = history + [{"role": "user", "content": message}]
        stream = client.chat.completions.create(
            model=OPENAI_MINI_MODEL, messages=messages, stream=True,
            **_generation_kwargs(1024, OPENAI_MINI_MODEL),
        )
        parts = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            parts.append(delta)
        print()
        history = (messages + [{"role": "assistant", "content": "".join(parts)}])[-6:]


def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Hợp đồng side lab retry mọi Exception, tối đa 1 + max_retries lần."""
    if max_retries < 0 or base_delay < 0:
        raise ValueError("max_retries và base_delay phải không âm")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2 ** attempt)


def batch_compare(prompts: list[str]) -> list[dict]:
    return [{**compare_models(prompt), "prompt": prompt} for prompt in prompts]


def format_comparison_table(results: list[dict]) -> str:
    headers = ["Prompt", "GPT-4o Response", "Mini Response", "GPT-4o Latency (s)", "Mini Latency (s)"]

    def brief(value):
        text = " ".join(str(value).split()).replace("|", "\\|")
        return text if len(text) <= 40 else text[:37] + "..."

    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for result in results:
        lines.append(" | ".join([
            brief(result["prompt"]), brief(result["gpt4o_response"]), brief(result["mini_response"]),
            f'{result["gpt4o_latency"]:.3f}', f'{result["mini_latency"]:.3f}',
        ]))
    return "\n".join(lines)


if __name__ == "__main__":
    prompt = "Explain the difference between temperature and top_p in one sentence."
    print(format_comparison_table(batch_compare([prompt])))
    streaming_chatbot()
