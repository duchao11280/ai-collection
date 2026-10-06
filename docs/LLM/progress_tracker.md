# Bảng theo dõi tiến độ

Đi kèm [learning_plan.md](learning_plan.md). Tick vào ô mỗi khi hoàn thành, ghi ngày thực tế để biết mình đang sớm hay trễ.

**Ngày bắt đầu:** `____-__-__`  ·  **Dự kiến kết thúc:** `____-__-__` (+16 tuần)

---

## Tuần 0 — Setup

- [x] Python 3.11 + `uv`
    + Dùng conda, chờ conda ổn dịnh thì có thể bật lại bản nén bằng conda config --remove-key repodata_use_zst.

- [x] Docker Desktop + WSL2
- [ ] API key provider + budget cap đã đặt
- [x] Ollama chạy được 1 model local — hướng dẫn: [ollama_setup.md](ollama_setup.md)
- [x] Repo `llm-lab` khởi tạo theo cấu trúc đề xuất
- [x] "Hello LLM" chạy được từ CLI

---

## Giai đoạn 1 — Nền tảng

### Tuần 1
- [ ] D1 — Token, context window, pricing
- [ ] D2 — Prompt engineering & structured output
- [ ] D3 — LangChain LCEL / Runnable
- [ ] D4 — Embeddings & semantic search bằng numpy
- [ ] D5 — Chroma vector DB
- [ ] Cuối tuần — Mini project semantic search

---

## Giai đoạn 2 — RAG

### Tuần 2 — Ingestion & Chunking
- [ ] D1 — Loaders (PDF/DOCX/HTML/OCR)
- [ ] D2 — Cleaning & dedupe
- [ ] D3 — So sánh 5 chiến lược chunking
- [ ] D4 — Thiết kế metadata schema (có `tenant_id`, `acl_groups`)
- [ ] D5 — Ingestion pipeline idempotent + incremental
- [ ] Cuối tuần — Ingest corpus thật, ghi số liệu

### Tuần 3 — Embedding, Indexing & Retrieval
- [ ] D1 — So sánh 3 embedding model
- [ ] D2 — HNSW / IVF-PQ internals
- [ ] D3 — Qdrant trên Docker
- [ ] D4 — Dense / BM25 / Hybrid + RRF
- [ ] D5 — Query transformation (rewrite, HyDE, multi-query)
- [ ] Cuối tuần — Benchmark retrieval (Recall@k, MRR, nDCG)

### Tuần 4 — Re-ranking, Generation & Eval
- [ ] D1 — Cross-encoder re-ranking 2 tầng
- [ ] D2 — Context assembly & citation
- [ ] D3 — Grounded generation + hành vi "không biết"
- [ ] D4 — Golden dataset 50–100 câu
- [ ] D5 — Ragas / DeepEval harness
- [ ] Cuối tuần — **Báo cáo baseline**

### Tuần 5 — RAG nâng cao & Monitoring
- [ ] D1 — Failure analysis 20 case tệ nhất
- [ ] D2 — Contextual retrieval / parent-document
- [ ] D3 — Caching (embedding, semantic, prompt)
- [ ] D4 — Tracing với Langfuse
- [ ] D5 — Dashboard metric online
- [ ] 🏁 **Milestone 1 — RAG service v1 + báo cáo eval**

---

## Giai đoạn 3 — Agentic AI

### Tuần 6 — Tool use
- [ ] D1 — Thiết kế tool schema
- [ ] D2 — Viết 4–5 tool có validate
- [ ] D3 — ReAct loop tự code, không framework
- [ ] D4 — Error handling & circuit breaker
- [ ] D5 — LangGraph cơ bản
- [ ] Cuối tuần — Agent RAG + SQL

### Tuần 7 — Planning & Memory
- [ ] D1 — Plan-and-execute, reflection
- [ ] D2 — Checkpointer, human-in-the-loop, parallel branch
- [ ] D3 — Multi-agent supervisor
- [ ] D4 — Short-term & long-term memory
- [ ] D5 — Persistence Redis/Postgres
- [ ] Cuối tuần — Agent resume được sau crash

### Tuần 8 — Guardrails
- [ ] D1 — Input guardrails + tự tấn công agent
- [ ] D2 — Output guardrails & PII redaction
- [ ] D3 — Tool authorization & human approval
- [ ] D4 — Trajectory eval, task success rate
- [ ] D5 — Budget cap & loop detection
- [ ] 🏁 **Milestone 2 — Agent service + eval suite + guardrails**

---

## Giai đoạn 4 — Inference & chi phí

### Tuần 9 — Nền tảng inference
- [ ] D1 — Prefill/decode, KV cache, tính VRAM
- [ ] D2 — Script đo TTFT / TPOT / throughput
- [ ] D3 — Quantization + đo chất lượng bằng golden dataset
- [ ] D4 — vLLM / TensorRT-LLM / SGLang
- [ ] D5 — Benchmark sweep batch & concurrency
- [ ] Cuối tuần — Báo cáo benchmark

```
total duration:       45.79458s
load duration:        7.2810946s
prompt eval count:    20 token(s)
prompt eval duration: 190.92ms
prompt eval rate:     104.76 tokens/s
eval count:           1850 token(s)
eval duration:        38.30783s
eval rate:            48.29 tokens/s
```

### Tuần 10 — NIM, ATOM & cost
- [ ] D1 — Kiến trúc NVIDIA NIM
- [ ] D2 — Chạy NIM và cắm vào RAG service
- [ ] D3 — Rebellions ATOM / RBLN SDK
- [ ] D4 — Cost model & điểm hoà vốn self-host vs API
- [ ] D5 — Model routing, cascade, semantic cache
- [ ] 🏁 **Milestone 3 — Báo cáo chọn hạ tầng inference**

---

## Giai đoạn 5 — Production

### Tuần 11 — API & Service
- [ ] D1 — FastAPI + SSE streaming
- [ ] D2 — Concurrency & queue
- [ ] D3 — Dockerfile multi-stage, non-root
- [ ] D4 — docker compose full stack
- [ ] D5 — K8s / Cloud Run cơ bản
- [ ] Cuối tuần — Deploy lên cloud thật, có HTTPS

### Tuần 12 — CI/CD & Observability
- [ ] D1 — Test suite + mock LLM
- [ ] D2 — CI pipeline (lint, test, build, scan)
- [ ] D3 — CD + eval gate + rollback
- [ ] D4 — Log / metric / trace (OpenTelemetry)
- [ ] D5 — LLM metric: token, cost, prompt version, drift
- [ ] Cuối tuần — Debug lỗi chỉ bằng dashboard

### Tuần 13 — Reliability
- [ ] D1 — SLI/SLO + error budget
- [ ] D2 — Retry, fallback, circuit breaker
- [ ] D3 — Load test & capacity planning
- [ ] D4 — Graceful degradation
- [ ] D5 — Runbook, alert, postmortem template
- [ ] 🏁 **Milestone 4 — Chaos test không sập**

---

## Giai đoạn 6 — Security & Scale

### Tuần 14 — AuthN/AuthZ & Data protection
- [ ] D1 — API key, OAuth2/OIDC, JWT, mTLS
- [ ] D2 — RBAC + ACL filter ở tầng retrieval
- [ ] D3 — Secrets management & rotation
- [ ] D4 — PII redaction, encryption, retention policy
- [ ] D5 — Chống prompt injection (cả indirect qua tài liệu)

### Tuần 15 — Rate limit, Audit & Availability
- [ ] D1 — Rate limiting & quota theo tenant
- [ ] D2 — Abuse prevention & moderation
- [ ] D3 — Audit log bất biến
- [ ] D4 — Scale ngang, sharding vector DB
- [ ] D5 — Multi-AZ, failover, diễn tập restore
- [ ] 🏁 **Milestone 5 — Threat model + security checklist**

---

## Giai đoạn 7 — Capstone

### Tuần 16
- [ ] D1–D2 — Design doc (kiến trúc, trade-off, SLO, chi phí)
- [ ] D3–D4 — Tích hợp toàn hệ thống
- [ ] D5 — Load test + full eval + cost report
- [ ] 🏁 **Milestone 6 — Demo + blog post + retrospective**

---

## Bảng số liệu (điền dần)

Ghi lại để thấy mình thực sự cải thiện, không phải cảm giác.

| Mốc | Recall@5 | nDCG@10 | Faithfulness | Answer relevancy | Latency p95 | Cost / 1k query |
|---|---|---|---|---|---|---|
| Baseline (T4) | | | | | | |
| Sau re-rank (T4) | | | | | | |
| Sau advanced RAG (T5) | | | | | | |
| Sau tối ưu inference (T10) | | | | | | |
| Capstone (T16) | | | | | | |

| Mốc | Task success rate | Số bước TB | Cost / task | Guardrail block rate |
|---|---|---|---|---|
| Agent v1 (T6) | | | | |
| Agent + guardrails (T8) | | | | |
| Capstone (T16) | | | | |

---

## Nhật ký tuần

Mỗi cuối tuần viết 4 dòng, không hơn:

```markdown
### Tuần X (YYYY-MM-DD → YYYY-MM-DD)
- Hoàn thành: __/6 buổi
- Output tuần này:
- Số liệu thay đổi:
- Điều chỉnh cho tuần sau:
```
