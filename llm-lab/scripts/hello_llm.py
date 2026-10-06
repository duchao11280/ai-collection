"""Tuần 0: gọi LLM local qua Ollama để xác nhận môi trường chạy được."""

import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")


def split_thinking(message: dict) -> tuple[str, str]:
    """Trả về (thinking, answer).

    Ollama mới trả thinking ở field riêng `message.thinking`;
    bản cũ nhét vào `content` dạng <think>...</think>.
    """
    content = message.get("content", "")
    thinking = message.get("thinking", "")
    if "</think>" in content:
        head, content = content.split("</think>", 1)
        thinking = thinking or head.replace("<think>", "")
    return thinking.strip(), content.strip()


def main() -> None:
    start = time.perf_counter()
    resp = httpx.post(
        f"{BASE_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Xin chào! Giới thiệu bản thân trong 1 câu."}],
            "stream": False,
            "think": True,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.perf_counter() - start

    thinking, answer = split_thinking(data["message"])
    print(f"--- thinking ({len(thinking)} ký tự) ---\n{thinking}\n")
    print(f"--- answer ---\n{answer}")
    print(
        f"\n[{MODEL}] prompt_tokens={data.get('prompt_eval_count')} "
        f"output_tokens={data.get('eval_count')} (gồm cả thinking) latency={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
