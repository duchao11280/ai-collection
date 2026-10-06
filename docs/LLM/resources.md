# Tài liệu học theo chủ đề

Tài liệu đi kèm [learning_plan.md](learning_plan.md). Nguyên tắc: **docs chính thức trước, blog sau, paper cuối cùng**.
Mỗi ngày chỉ đọc 1 mục — danh sách này là để tra cứu, không phải để đọc hết.

!!! warning "Link có thể đổi"
    Hệ sinh thái LLM thay đổi rất nhanh, URL và tên API có thể khác đi. Nếu một link chết, tìm lại từ trang chủ của dự án thay vì bỏ qua chủ đề.

---

## Tuần 1 — Nền tảng & LangChain

| Loại | Nguồn |
|---|---|
| Docs | [Anthropic Claude Docs](https://docs.anthropic.com/) — prompt engineering, tool use, prompt caching, token counting |
| Docs | [LangChain (Python)](https://python.langchain.com/) — bắt đầu từ mục *Introduction* và *LCEL* |
| Docs | [Ollama](https://ollama.com/) — chạy model local, tiết kiệm chi phí khi luyện tập. Hướng dẫn cài & chọn model: [ollama_setup.md](ollama_setup.md) |
| Docs | [Sentence Transformers](https://sbert.net/) — embedding, cosine similarity, normalize |
| Tham khảo | [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — chọn embedding model, nhớ lọc theo đa ngữ |
| Docs | [Chroma](https://docs.trychroma.com/) — vector DB đơn giản nhất để bắt đầu |

## Tuần 2 — Ingestion & Chunking

| Loại | Nguồn |
|---|---|
| Lib | [PyMuPDF](https://pymupdf.readthedocs.io/), [pdfplumber](https://github.com/jsvine/pdfplumber) — trích xuất PDF, bảng biểu |
| Lib | [unstructured](https://github.com/Unstructured-IO/unstructured) — loader đa định dạng |
| Docs | LangChain *Text Splitters* — recursive, markdown header, semantic chunker |
| Blog | [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — kỹ thuật chèn ngữ cảnh vào chunk |

## Tuần 3 — Indexing & Retrieval

| Loại | Nguồn |
|---|---|
| Docs | [Qdrant](https://qdrant.tech/documentation/) — HNSW, payload filter, hybrid search |
| Docs | [pgvector](https://github.com/pgvector/pgvector) — khi muốn dùng luôn Postgres sẵn có |
| Docs | [Milvus](https://milvus.io/docs), [Weaviate](https://weaviate.io/developers/weaviate) — tham khảo để so sánh |
| Docs | [FAISS](https://faiss.ai/) — hiểu index internals (Flat, IVF, PQ, HNSW) |
| Paper | *HyDE* — [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) |
| Paper | *ColBERT* — [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) — late interaction retrieval |

## Tuần 4–5 — Re-ranking, Evaluation & Monitoring

| Loại | Nguồn |
|---|---|
| Model | `BAAI/bge-reranker-v2-m3` trên Hugging Face — cross-encoder đa ngữ |
| Docs | [Ragas](https://docs.ragas.io/) — faithfulness, context precision/recall |
| Docs | [DeepEval](https://github.com/confident-ai/deepeval) — eval chạy như unit test |
| Docs | [Langfuse](https://langfuse.com/docs) — tracing, cost tracking, self-host bằng Docker |
| Docs | [LangSmith](https://docs.smith.langchain.com/) — nếu dùng hệ sinh thái LangChain |
| Paper | *RAG* gốc — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| Paper | *Lost in the Middle* — [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) |
| Paper | *Self-RAG* — [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) |
| Paper | *RAPTOR* — [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) |

## Tuần 6–8 — Agentic AI

| Loại | Nguồn |
|---|---|
| Blog | [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — **đọc trước khi viết agent đầu tiên** |
| Docs | [LangGraph](https://langchain-ai.github.io/langgraph/) — state machine, checkpointer, human-in-the-loop |
| Docs | [Model Context Protocol](https://modelcontextprotocol.io/) — chuẩn kết nối tool/data cho agent |
| Docs | [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails), [Guardrails AI](https://www.guardrailsai.com/docs) |
| Docs | [Pydantic](https://docs.pydantic.dev/) — validate input/output của tool |
| Security | [OWASP Top 10 for LLM Applications](https://genai.owasp.org/) — nền tảng cho tuần 8 và tuần 14 |
| Paper | *ReAct* — [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) |
| Paper | *Reflexion* — [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) |
| Paper | *Toolformer* — [arXiv:2302.04761](https://arxiv.org/abs/2302.04761) |

## Tuần 9–10 — Inference, NIM & ATOM

| Loại | Nguồn |
|---|---|
| Docs | [vLLM](https://docs.vllm.ai/) — PagedAttention, continuous batching, benchmark script có sẵn |
| Docs | [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) — engine tối ưu cho GPU NVIDIA |
| Docs | [SGLang](https://docs.sglang.ai/) — RadixAttention, prefix caching |
| Docs | [NVIDIA NIM](https://docs.nvidia.com/nim/) — microservice inference đóng gói sẵn |
| Thử nghiệm | [build.nvidia.com](https://build.nvidia.com/) — dùng thử NIM API không cần GPU riêng |
| Docs | [Rebellions](https://rebellions.ai/) và RBLN SDK docs (`docs.rbln.ai`) — ATOM NPU, `rebel-compiler`, `vllm-rbln` |
| Paper | *vLLM / PagedAttention* — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) |
| Paper | *AWQ* — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978), *GPTQ* — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |

## Tuần 11–13 — Production & Reliability

| Loại | Nguồn |
|---|---|
| Docs | [FastAPI](https://fastapi.tiangolo.com/) — async, streaming, dependency injection |
| Docs | [The Twelve-Factor App](https://12factor.net/) — nguyên tắc config & deploy |
| Docs | [Docker](https://docs.docker.com/), [Kubernetes](https://kubernetes.io/docs/home/) |
| Docs | [GitHub Actions](https://docs.github.com/actions) |
| Docs | [OpenTelemetry — GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) |
| Docs | [Prometheus](https://prometheus.io/docs/), [Grafana](https://grafana.com/docs/) |
| Docs | [k6](https://grafana.com/docs/k6/latest/) hoặc [Locust](https://docs.locust.io/) — load testing |
| Sách | [Google SRE Book](https://sre.google/books/) — chương SLO, alerting, postmortem |

## Tuần 14–15 — Security, Scale & Availability

| Loại | Nguồn |
|---|---|
| Security | [OWASP Top 10 for LLM Apps](https://genai.owasp.org/) — prompt injection, insecure output handling, data leakage |
| Security | [OWASP API Security Top 10](https://owasp.org/www-project-api-security/) |
| Framework | [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) |
| Docs | [HashiCorp Vault](https://developer.hashicorp.com/vault/docs) — secrets management |
| Lib | [Microsoft Presidio](https://microsoft.github.io/presidio/) — phát hiện & che PII |
| Docs | [Redis rate limiting patterns](https://redis.io/docs/latest/) — token bucket, sliding window |

---

## Khoá học / nội dung dài (tuỳ chọn)

Chỉ dùng khi thấy hổng kiến thức nền, không thay thế phần thực hành:

- **DeepLearning.AI** — các khoá ngắn về RAG, LangChain, agent, evaluation (miễn phí, mỗi khoá 1–2 giờ).
- **Hugging Face NLP Course** — nền tảng transformer, tokenizer, fine-tuning.
- **NVIDIA DLI** — các lab về inference optimization và triển khai với NIM/TensorRT-LLM.

---

## Cách chọn nguồn khi bí

1. Có docs chính thức không? → đọc docs.
2. Docs không nói rõ? → đọc source code của thư viện. Nhanh hơn tìm blog.
3. Vẫn chưa rõ? → tìm paper gốc, đọc phần *Method* và *Limitations*.
4. Cuối cùng mới đến blog và video — và luôn kiểm tra ngày đăng.
