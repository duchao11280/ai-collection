# Kế hoạch học LLM hằng ngày (16 tuần)

Kế hoạch này dành cho người **đã có nền ML/CV** và muốn chuyển sang xây dựng **hệ thống LLM chạy production**.
Mục tiêu cuối: tự tay build và vận hành được một hệ thống RAG + Agent có API, CI/CD, observability, guardrails và bảo mật.

!!! note "Triết lý"
    - **Build > Read.** Mỗi ngày phải có code chạy được, không chỉ đọc.
    - **Đo trước khi tối ưu.** Không có số liệu thì không gọi là "cải thiện".
    - **Mỗi tuần một output.** Một service, một report, hoặc một bài note trong `docs/LLM/`.

---

## 1. Nhịp độ hằng ngày

| Khung | Thời lượng | Nội dung |
|---|---|---|
| Đọc / xem | 15–20 phút | 1 tài liệu hoặc 1 phần docs chính thức. Không lan man. |
| Thực hành | 55–60 phút | Viết code theo nhiệm vụ của ngày hôm đó. |
| Ghi chú | 10–15 phút | Tóm tắt 5–10 dòng vào note + commit. |

- **5 buổi/tuần × 90 phút** + **1 buổi cuối tuần × 2–3 giờ** (tích hợp, benchmark, viết report).
- Một ngày trong tuần là ngày nghỉ. Nghỉ thật, đừng dồn bài.
- **Luật bất di bất dịch:** mỗi ngày commit ít nhất 1 thứ (code hoặc note). Chuỗi commit chính là thước đo kỷ luật.

### Khi bị lỡ nhịp

Đừng cố học bù. Đánh dấu ngày đó là *skipped* trong [progress_tracker.md](progress_tracker.md) rồi đi tiếp.
Nếu lỡ trên 3 ngày trong một tuần, hãy lặp lại tuần đó thay vì nhảy cóc — các tuần sau phụ thuộc trực tiếp vào output của tuần trước.

---

## 2. Bản đồ lộ trình

| Giai đoạn | Tuần | Mục tiêu | Output bắt buộc |
|---|---|---|---|
| 0. Setup | Tuần 0 (2–3 ngày) | Môi trường, API key, repo skeleton | Repo `llm-lab` chạy được "hello LLM" |
| 1. Nền tảng | 1 | LLM cơ bản, embedding, LangChain, vector DB | Semantic search trên tài liệu của bạn |
| 2. RAG | 2–5 | Ingestion → chunking → index → retrieval → rerank → eval → monitor | **M1:** RAG service + báo cáo eval |
| 3. Agentic | 6–8 | Tool use, planning, orchestration, memory, guardrails | **M2:** Agent service + eval suite |
| 4. Inference | 9–10 | Tối ưu hiệu năng & chi phí, vLLM, NVIDIA NIM, Rebellions ATOM | **M3:** Benchmark + báo cáo chọn hạ tầng |
| 5. Production | 11–13 | API, container, CI/CD, observability, reliability | **M4:** Stack deploy được + dashboard + chaos test |
| 6. Security & Scale | 14–15 | AuthN/AuthZ, secrets, data protection, rate limit, audit | **M5:** Threat model + security checklist |
| 7. Capstone | 16 | Ráp tất cả thành một sản phẩm | **M6:** Design doc + demo + blog post |

---

## 3. Tuần 0 — Chuẩn bị môi trường (2–3 ngày)

Bạn đang dùng Windows 11, nên chuẩn bị sẵn cả hai đường: native Windows cho phần app, WSL2 cho phần serving/GPU.

- [x] Python 3.11 + `uv` (hoặc Poetry) để quản lý môi trường.
    + Dùng conda llm
- [x] Docker Desktop + WSL2 backend (bắt buộc cho Qdrant, Langfuse, vLLM, NIM ở các tuần sau).
- [x] Git + tài khoản GitHub (dùng cho CI/CD ở tuần 12).
- [x] API key: chọn **1 provider trả phí** (Anthropic Claude hoặc tương đương) + **1 đường local miễn phí** (Ollama + `qwen3:4b`, xem [ollama_setup.md](ollama_setup.md) để chọn model theo cấu hình máy) để học mà không đốt tiền.
    + Dùng Local qwen3:4b
- [ ] Đặt **budget cap** trên dashboard provider ngay hôm nay. Đây không phải tùy chọn.
    + Không áp dụng
- [ ] Nếu không có GPU: tạo sẵn tài khoản Google Colab / Kaggle / RunPod / Modal cho tuần 9–10.

### Cấu trúc repo thực hành đề xuất

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
└── docs/                # note hằng ngày, design doc, report
```

---

## 4. Giai đoạn 1 — Nền tảng (Tuần 1)

**Mục tiêu tuần:** hiểu LLM ở mức API, nắm được embedding và tự làm được semantic search.

**Tài liệu lý thuyết:** [Tuần 1 — Lý thuyết nền tảng](week_01/index.md) (mỗi ngày một bài, kèm câu hỏi tự kiểm tra và đáp án).

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Cơ chế LLM | Token, context window, temperature/top_p, stop sequence, pricing. Viết script gọi API + đếm token + tính chi phí thực của 20 request. |
| D2 | Prompt engineering | System prompt, few-shot, structured output (JSON schema / tool use), prompt caching. Ép model trả về JSON đúng schema 20/20 lần. |
| D3 | LangChain core | `Runnable`, LCEL, toán tử pipe, `PromptTemplate`, output parser, `.invoke()` vs `.stream()` vs `.batch()`. Build chain tóm tắt có streaming. |
| D4 | Embeddings | Cosine similarity, chuẩn hóa vector, chọn model đa ngữ cho tiếng Việt (`bge-m3`, `multilingual-e5`, `text-embedding-3`). Tự code semantic search bằng numpy trên ~200 đoạn văn. |
| D5 | Vector DB 101 | Chroma local: upsert, query, metadata filter, persistence. So sánh kết quả với numpy hôm trước — phải trùng nhau. |
| Cuối tuần | Mini project | Semantic search trên tài liệu tiếng Việt thật của bạn (300–500 đoạn). Đo: thời gian index, latency query, chi phí embedding. |

**Tự kiểm tra cuối tuần:**

- [ ] Giải thích được vì sao cùng một câu hỏi lại tốn số token khác nhau giữa 2 model.
- [ ] Viết được một LCEL chain 3 bước mà không cần copy-paste.
- [ ] Trả lời được: embedding 1024 chiều cho 100k chunk tốn bao nhiêu RAM?

---

## 5. Giai đoạn 2 — RAG end-to-end (Tuần 2–5)

### Tuần 2 — Data ingestion & Chunking

**Mục tiêu:** biến đống file lộn xộn thành corpus sạch, có metadata, và **cập nhật tăng dần được (incremental)**.

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Loaders | PDF (PyMuPDF, pdfplumber, `unstructured`), DOCX, HTML, Markdown. PDF scan → OCR (đây là chỗ nền CV của bạn có lợi thế). |
| D2 | Cleaning | Chuẩn hóa Unicode NFC cho tiếng Việt, bỏ header/footer lặp, xử lý bảng, khử trùng lặp bằng hash / MinHash. |
| D3 | Chunking | So sánh 5 chiến lược: fixed-size, recursive, markdown-header, sentence-window, parent-document. Vẽ phân bố độ dài chunk. |
| D4 | Metadata | Thiết kế schema: `doc_id`, `source`, `page`, `section`, `created_at`, `tenant_id`, `acl_groups`. Thêm ACL **ngay từ bây giờ** — tuần 14 sẽ cần. |
| D5 | Pipeline | Viết CLI ingestion: idempotent (chạy 2 lần không nhân đôi dữ liệu), incremental theo content hash, có log số liệu. |
| Cuối tuần | Chạy thật | Ingest toàn bộ corpus. Ghi lại: số doc, số chunk, token trung bình/chunk, tổng chi phí embedding, thời gian chạy. |

!!! warning "Bẫy phổ biến"
    Chunking sai là nguyên nhân số 1 của RAG kém, nhưng lại hay bị đổ lỗi cho model. Nếu câu trả lời bị cụt ý, hãy xem lại chunk trước khi đổi LLM.

### Tuần 3 — Embedding, Indexing & Retrieval

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Chọn embedding model | So sánh 3 model trên chính dữ liệu của bạn bằng 30–50 câu hỏi vàng. Tiêu chí: Recall@5, latency, chi phí, hỗ trợ tiếng Việt. |
| D2 | Index internals | HNSW (`M`, `efConstruction`, `efSearch`), IVF-PQ, Flat. Vẽ đường cong recall vs latency khi đổi `efSearch`. |
| D3 | Vector DB thật | Chạy Qdrant bằng Docker: collection, payload index, filter, snapshot. So sánh với `pgvector` về mặt vận hành. |
| D4 | Retrieval strategies | Dense vs BM25 sparse vs hybrid + RRF fusion. MMR để tăng độ đa dạng. Đo từng cái. |
| D5 | Query transformation | Query rewriting, HyDE, multi-query, decomposition, routing theo metadata. Xem cái nào thực sự giúp *dữ liệu của bạn*. |
| Cuối tuần | Benchmark | Script benchmark retrieval tái lập được: Recall@k, MRR, nDCG@10, latency p95. Lưu kết quả ra file để so sánh về sau. |

### Tuần 4 — Re-ranking, Generation & Evaluation

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Re-ranking | Cross-encoder (`bge-reranker-v2-m3`, Cohere Rerank). Kiến trúc 2 tầng: lấy top-50 → rerank → top-5. Đo cả mức tăng chất lượng lẫn mức tăng latency. |
| D2 | Context assembly | Hiện tượng *lost-in-the-middle*, thứ tự chunk, khử trùng lặp, ngân sách token, format trích dẫn. |
| D3 | Grounded generation | Bắt model trích dẫn nguồn, và bắt nó **nói "không biết"** khi context không đủ. Đây là tính năng, không phải lỗi. |
| D4 | Golden dataset | Tự tay viết 50–100 cặp Q/A từ corpus thật, gồm cả câu *không trả lời được* để test hallucination. Không có bước này thì mọi thứ sau đều là cảm tính. |
| D5 | Eval framework | Ragas / DeepEval: context precision, context recall, faithfulness, answer relevancy. Chạy được bằng một lệnh duy nhất. |
| Cuối tuần | Báo cáo baseline | Chạy full eval, ghi số baseline vào bảng. Mọi thay đổi từ tuần sau phải so với bảng này. |

### Tuần 5 — RAG nâng cao & Monitoring

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Failure analysis | Lấy 20 câu trả lời tệ nhất, phân loại nguyên nhân: retrieval miss / chunk hỏng / rerank sai / generation bịa. Sửa đúng nguyên nhân, đừng sửa bừa. |
| D2 | Advanced patterns | Parent-document retriever, contextual retrieval (chèn tóm tắt ngữ cảnh vào đầu chunk), summary index / RAPTOR. |
| D3 | Caching | Embedding cache, semantic cache (cache theo độ tương tự), prompt caching của provider. Đo phần trăm chi phí tiết kiệm được. |
| D4 | Tracing | Langfuse (self-host bằng Docker) hoặc LangSmith: trace, span, session, chi phí/request, feedback của người dùng. |
| D5 | Dashboard | Metric online: latency p50/p95, retrieval hit rate, token & cost mỗi query, tỉ lệ trả lời "không biết", thumbs up/down. |
| Cuối tuần | **Milestone 1** | RAG service v1 hoàn chỉnh + báo cáo eval so với baseline tuần 4. Viết 1 bài note vào `docs/LLM/`. |

---

## 6. Giai đoạn 3 — Agentic AI (Tuần 6–8)

### Tuần 6 — Tool use & vòng lặp agent

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Tool calling | Thiết kế JSON schema cho tool. Chất lượng `description` quan trọng hơn code. Thử parallel tool calls. |
| D2 | Viết tool | 4–5 tool thật: search KB (dùng RAG tuần 5), truy vấn SQL, calculator, HTTP fetch, đọc file. Validate input bằng Pydantic. |
| D3 | ReAct từ số 0 | Tự code vòng lặp agent **không dùng framework**: loop, dispatch tool, điều kiện dừng, giới hạn số bước. Hiểu rồi mới dùng framework. |
| D4 | Error handling | Tool ném exception → trả lỗi có cấu trúc về cho model, retry có backoff, timeout, circuit breaker, fallback. |
| D5 | LangGraph | State, node, edge, conditional routing. Port agent ReAct hôm D3 sang LangGraph và so sánh sự khác biệt. |
| Cuối tuần | Mini project | Agent hỏi–đáp dữ liệu nội bộ dùng đồng thời RAG tool + SQL tool, và biết chọn đúng tool. |

### Tuần 7 — Planning, Orchestration & Memory

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Planning patterns | Plan-and-execute, task decomposition, reflection / self-critique. Đo xem pattern nào thực sự tăng tỉ lệ thành công, cái nào chỉ đốt token. |
| D2 | LangGraph nâng cao | Checkpointer, `interrupt` cho human-in-the-loop, subgraph, nhánh chạy song song. |
| D3 | Multi-agent | Supervisor/router + agent chuyên môn, cơ chế handoff. **Và quan trọng hơn:** khi nào KHÔNG nên dùng multi-agent. |
| D4 | Memory | Short-term (buffer, summary), long-term (vector memory, user profile), session state. Chính sách ghi nhớ: cái gì đáng lưu, cái gì nên quên. |
| D5 | Persistence | Checkpoint vào Redis/Postgres, resume sau crash, đảm bảo bước tool là idempotent. |
| Cuối tuần | Bài test khó | Agent chạy 10+ bước, kill process giữa chừng, khởi động lại và **chạy tiếp đúng chỗ đang dở**. |

### Tuần 8 — Guardrails & độ tin cậy của agent

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Input guardrails | Prompt injection, jailbreak, phát hiện PII. Đọc OWASP Top 10 for LLM Applications rồi **tự tấn công agent của mình**. |
| D2 | Output guardrails | Validate schema, kiểm tra groundedness, lọc nội dung độc hại, che PII. Thử NeMo Guardrails hoặc Guardrails AI. |
| D3 | Tool authorization | Allowlist tool theo vai trò, scope quyền, chế độ dry-run, **bắt buộc người duyệt** cho hành động phá hủy (xóa, gửi mail, thanh toán). |
| D4 | Agent evaluation | Trajectory eval, task success rate, độ chính xác gọi tool, số bước & chi phí trung bình mỗi task. |
| D5 | Kiểm soát chi phí | Giới hạn số bước, budget cap theo request, timeout, phát hiện vòng lặp vô hạn. |
| Cuối tuần | **Milestone 2** | Agent service + eval suite + guardrails. Báo cáo: tỉ lệ thành công, chi phí/task, số lần guardrail chặn. |

---

## 7. Giai đoạn 4 — Tối ưu inference & chi phí (Tuần 9–10)

### Tuần 9 — Nền tảng inference

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Cơ chế inference | Prefill vs decode, KV cache, công thức ước lượng VRAM (trọng số + KV cache + activation). Tính tay cho model 7B ở batch 32. |
| D2 | Metric đúng | TTFT, TPOT/ITL, throughput (tok/s), goodput, concurrency. Viết script đo — đừng tin con số quảng cáo. |
| D3 | Quantization | FP16/BF16, FP8, INT8, AWQ, GPTQ. Đo mức giảm chất lượng **bằng chính golden dataset tuần 4**, không dùng benchmark chung chung. |
| D4 | Serving engines | vLLM (PagedAttention, continuous batching), TensorRT-LLM, SGLang, speculative decoding. Đọc kiến trúc, hiểu vì sao nhanh. |
| D5 | Benchmark tay | Chạy vLLM (local GPU / Colab / RunPod), sweep batch size và concurrency, vẽ đường cong latency–throughput, tìm điểm gãy. |
| Cuối tuần | Report | Báo cáo benchmark: cấu hình nào cho throughput tốt nhất dưới ràng buộc TTFT p95 < 1s. |

### Tuần 10 — NVIDIA NIM, Rebellions ATOM & kinh tế học chi phí

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | NVIDIA NIM | Kiến trúc microservice đóng gói sẵn, container, API tương thích OpenAI, model profile theo GPU, và cách nó dựa trên TensorRT-LLM/vLLM bên dưới. |
| D2 | NIM thực hành | Chạy 1 NIM container (nếu có GPU) **hoặc** dùng endpoint trên `build.nvidia.com`. Đo TTFT/throughput, rồi cắm thẳng vào RAG service tuần 5 — chỉ cần đổi `base_url`. |
| D3 | Rebellions ATOM | NPU ATOM + RBLN SDK (`rebel-compiler`, `vllm-rbln`): luồng compile → deploy, danh sách model hỗ trợ, đặc điểm perf/watt so với GPU. |
| D4 | Cost model | Xây spreadsheet: $/1M token, điểm hòa vốn giữa self-host và API, GPU-hour, tỉ lệ sử dụng (utilization), chi phí lúc traffic thấp. |
| D5 | Routing & tiering | Model nhỏ trước → cascade lên model lớn khi cần, semantic cache, gom batch cho job offline, nén prompt. Đo số tiền tiết kiệm thật. |
| Cuối tuần | **Milestone 3** | Báo cáo "Chọn hạ tầng inference nào cho use case X" — có số liệu, có giả định, có khuyến nghị rõ ràng. |

!!! tip "Nếu không có GPU hoặc phần cứng ATOM"
    Đừng bỏ tuần này. Thay bằng: (1) benchmark trên Colab/RunPod thuê theo giờ, (2) đọc kỹ docs NIM và RBLN SDK rồi viết bản so sánh kiến trúc, (3) tập trung vào cost model và routing — phần này mang lại giá trị lớn nhất mà không cần phần cứng.
    ATOM là NPU chuyên cho inference của Rebellions; việc truy cập thường qua cloud partner hoặc chương trình dành cho nhà phát triển, nên hãy xác nhận điều kiện truy cập trước khi lên lịch thực hành.

---

## 8. Giai đoạn 5 — Hệ thống production (Tuần 11–13)

### Tuần 11 — API & Service

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | FastAPI | Endpoint async, streaming qua SSE, Pydantic model, error model thống nhất, versioning `/v1`. |
| D2 | Concurrency | Async pool, backpressure, hàng đợi cho job dài (Redis/Celery). Không để một request LLM chậm làm chết cả service. |
| D3 | Container | Multi-stage Dockerfile, chạy non-root, healthcheck, cấu hình qua env (12-factor), image nhẹ. |
| D4 | Compose stack | `docker compose` gồm: api + Qdrant + Redis + Postgres + Langfuse. Một lệnh là cả hệ thống chạy. |
| D5 | Orchestration | K8s cơ bản: Deployment, Service, HPA, readiness/liveness probe, resource request/limit. Hoặc chọn Cloud Run/ECS nếu muốn đơn giản hơn. |
| Cuối tuần | Deploy | Đưa stack lên một VPS hoặc cloud thật, truy cập được từ internet qua HTTPS. |

### Tuần 12 — CI/CD & Observability

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Testing | Unit test cho chunker/retriever, mock LLM để test nhanh và rẻ, contract test, snapshot test cho prompt. |
| D2 | CI | GitHub Actions: lint → test → build image → scan bảo mật (trivy) → push registry. |
| D3 | CD + eval gate | Staging → prod, migration, canary/blue-green, rollback. **Eval gate:** điểm eval tụt quá ngưỡng thì chặn deploy. |
| D4 | Observability | Log có cấu trúc kèm `trace_id`, metric Prometheus, trace OpenTelemetry (GenAI semantic conventions), dashboard Grafana. |
| D5 | LLM-specific | Token & chi phí theo request/tenant, version prompt và model trong trace, điểm eval online, phát hiện drift. |
| Cuối tuần | Kiểm chứng | Gây lỗi có chủ đích, rồi dùng **chỉ dashboard và log** để tìm nguyên nhân. Không tìm được nghĩa là observability chưa đủ. |

### Tuần 13 — Reliability Engineering

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | SLI/SLO | Định nghĩa SLI cho LLM service: TTFT p95, tỉ lệ lỗi, và cả *chất lượng câu trả lời*. Đặt SLO + error budget. |
| D2 | Failure modes | Provider sập, 429 rate limit, timeout, model trả rác. Xử lý: retry + jitter, fallback sang model khác, circuit breaker. |
| D3 | Load testing | k6 hoặc Locust: tìm điểm gãy, capacity planning, xác định cần bao nhiêu replica cho X RPS. |
| D4 | Graceful degradation | Chế độ chỉ-dùng-cache, hạ xuống model nhỏ, xếp hàng, load shedding. Chậm còn hơn chết. |
| D5 | Incident response | Runbook, alert rule (alert theo triệu chứng chứ không theo nguyên nhân), template postmortem. |
| Cuối tuần | **Milestone 4** | Chaos test: tắt vector DB, giả lập provider trả 429 liên tục. Hệ thống phải xuống cấp có kiểm soát, không được sập. |

---

## 9. Giai đoạn 6 — Security, Scalability & Availability (Tuần 14–15)

### Tuần 14 — AuthN/AuthZ & bảo vệ dữ liệu

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Authentication | API key có scope, OAuth2/OIDC, verify JWT, xác thực service-to-service (mTLS). |
| D2 | Authorization | RBAC, cô lập multi-tenant, và **ACL ở tầng retrieval**: lọc document theo quyền *trước khi* đưa vào context. Đây là lỗ hổng rò rỉ dữ liệu phổ biến nhất của RAG. |
| D3 | Secrets | Từ `.env` lên vault (HashiCorp Vault / cloud secret manager), xoay vòng key, chặn secret lọt vào log và trace. |
| D4 | Data protection | Phát hiện & che PII trước khi gửi lên LLM, mã hóa at-rest/in-transit, chính sách lưu trữ (retention), bật zero-retention phía provider. |
| D5 | Prompt injection | Phòng thủ nhiều lớp: tách rõ instruction và data, sandbox tool, escape output (LLM có thể sinh ra XSS), và **indirect injection qua chính tài liệu trong RAG**. |

### Tuần 15 — Rate limiting, Audit, Scale & Availability

| Ngày | Chủ đề | Nhiệm vụ thực hành |
|---|---|---|
| D1 | Rate limiting | Token bucket trên Redis, quota theo tenant, trần chi phí, trả 429 đúng chuẩn kèm `Retry-After`. |
| D2 | Abuse prevention | Phát hiện bất thường, content moderation, chặn theo user/IP, bảo vệ khỏi bị đốt token. |
| D3 | Audit logging | Ai làm gì, lúc nào, với dữ liệu nào. Log bất biến, an toàn PII, xuất được báo cáo tuân thủ. |
| D4 | Scalability | API stateless, scale ngang, sharding/replication vector DB, read replica, cache tầng biên. |
| D5 | Availability | Multi-AZ, failover giữa nhiều provider, backup & DR, định nghĩa RTO/RPO. **Diễn tập restore thật một lần.** |
| Cuối tuần | **Milestone 5** | Threat model + security checklist theo OWASP LLM Top 10, tự đánh giá hệ thống và liệt kê lỗ hổng còn lại. |

---

## 10. Tuần 16 — Capstone

| Ngày | Nhiệm vụ |
|---|---|
| D1–D2 | Chọn một bài toán thật (trợ lý tài liệu nội bộ, QA cho sổ tay kỹ thuật, agent phân tích báo cáo...). Viết **design doc**: yêu cầu, kiến trúc, lựa chọn & đánh đổi, SLO, ước tính chi phí. |
| D3–D4 | Ráp toàn bộ: RAG + agent + guardrails + API + CI/CD + observability + security. Dùng lại code 15 tuần trước, không viết lại từ đầu. |
| D5 | Load test + full eval + báo cáo chi phí thực tế trên 1000 query. |
| Cuối tuần | **Milestone 6:** demo, viết bài tổng kết vào `docs/LLM/`, và **retrospective**: cái gì hiệu quả, cái gì phí thời gian, học tiếp gì. |

---

## 11. Sau 16 tuần — duy trì

Học xong không có nghĩa là dừng. Nhịp duy trì hằng tuần:

- **2 buổi/tuần**: cải tiến hệ thống capstone — mỗi lần một thay đổi, luôn đo bằng eval.
- **1 buổi/tuần**: đọc 1 paper hoặc engineering blog, viết tóm tắt 10 dòng.
- **Hằng tháng**: cập nhật golden dataset bằng các câu hỏi thất bại thu được từ production.
- **Hằng quý**: đánh giá lại lựa chọn model/hạ tầng — lĩnh vực này giá và năng lực thay đổi rất nhanh.

---

## 12. Template ghi chú hằng ngày

Copy block này vào note mỗi ngày, viết tối đa 10 dòng:

```markdown
## YYYY-MM-DD — Tuần X / Ngày Y — <chủ đề>

**Đã làm:** (1–2 dòng)
**Học được:** (2–3 gạch đầu dòng, viết bằng lời của mình)
**Số liệu:** (latency / recall / cost / token — nếu có)
**Vướng mắc:** (cái gì chưa hiểu, sẽ tra lại khi nào)
**Ngày mai:** (1 dòng, cụ thể)
```

---

## 13. Tài liệu & theo dõi tiến độ

- Danh sách tài liệu học theo từng chủ đề: [resources.md](resources.md)
- Bảng theo dõi tiến độ & milestone: [progress_tracker.md](progress_tracker.md)
- Ghi chú nền tảng về RAG / LangChain: [index.md](index.md)
