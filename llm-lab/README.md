# llm-lab

Repo thực hành cho [learning plan LLM 16 tuần](../docs/LLM/learning_plan.md).

## Cấu trúc

```text
llm-lab/
├── apps/api/            # FastAPI service (tuần 11+)
├── libs/
│   ├── ingestion/       # loader, cleaner, chunker (tuần 2)
│   ├── retrieval/       # embedder, index, retriever, reranker (tuần 3-4)
│   ├── agents/          # tools, graph, memory (tuần 6-8)
│   └── guardrails/      # input/output validation (tuần 8, 14)
├── evals/
│   ├── golden/          # bộ câu hỏi vàng (tuần 4)
│   └── suites/          # eval harness chạy được trong CI
├── benchmarks/          # script đo TTFT/throughput (tuần 9-10)
├── infra/               # docker-compose, k8s manifest, CI workflow
├── notebooks/           # thử nghiệm nhanh, KHÔNG để code production ở đây
├── scripts/             # script chạy lẻ (hello_llm, ...)
└── docs/                # note hằng ngày, design doc, report
```

## Bắt đầu

```bash
conda activate llm          # Python 3.11
cd llm-lab
pip install -e ".[dev]"
cp .env.example .env
ollama pull qwen3:4b
python scripts/hello_llm.py
```
