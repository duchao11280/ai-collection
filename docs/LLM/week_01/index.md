# Tuần 1 — Lý thuyết nền tảng

Tài liệu lý thuyết đi kèm **Giai đoạn 1 — Nền tảng** trong [kế hoạch học 16 tuần](../learning_plan.md).
Mỗi ngày đọc bài tương ứng trong 15–20 phút **trước khi** code, rồi quay lại tra cứu khi làm bài thực hành.

!!! note "Mục tiêu của tuần"
    Hiểu LLM (Large Language Model — mô hình ngôn ngữ lớn) ở mức API, nắm được embedding và tự làm được semantic search (tìm kiếm theo ngữ nghĩa) trên tài liệu tiếng Việt của chính bạn.

---

## 1. Lịch đọc

| Ngày | Bài đọc | Sau khi đọc phải trả lời được | Bài thực hành trong kế hoạch |
|---|---|---|---|
| Ngày 1 | [Cơ chế hoạt động của LLM](day_1_llm_mechanics.md) | Token là gì, vì sao tiếng Việt tốn token hơn, temperature tác động vào đâu, vì sao token đầu ra đắt hơn | Gọi API, đếm token, tính chi phí 20 request |
| Ngày 2 | [Prompt engineering và structured output](day_2_prompt_engineering.md) | Vì sao constrained decoding đảm bảo được định dạng, prompt caching hoạt động theo tiền tố thế nào | Ép model trả JSON đúng schema 20/20 lần |
| Ngày 3 | [LangChain core: Runnable và LCEL](day_3_langchain_core.md) | Kiểu dữ liệu chảy qua từng bước của chain, khác biệt bản chất giữa `invoke`, `stream`, `batch` | Chain tóm tắt có streaming |
| Ngày 4 | [Embeddings](day_4_embeddings.md) | Vì sao chuẩn hóa vector làm cosine, dot product, L2 cho cùng thứ hạng; chọn model cho tiếng Việt | Semantic search bằng numpy trên khoảng 200 đoạn |
| Ngày 5 | [Vector database 101 với Chroma](day_5_vector_database.md) | Khoảng cách Chroma trả về nghĩa là gì, ID tất định và upsert để chạy lại không nhân đôi dữ liệu | So sánh Chroma với numpy — phải trùng |
| Cuối tuần | [Mini project và ôn tập](weekend_review.md) | Đo thời gian lập chỉ mục, latency theo phân vị, chi phí embedding; đáp án câu hỏi tự kiểm tra | Semantic search trên 300–500 đoạn thật |

---

## 2. Bức tranh tổng thể

Năm ngày học năm mảnh ghép. Tuần 2–5 sẽ ráp chúng thành RAG (Retrieval-Augmented Generation — sinh câu trả lời có truy xuất tài liệu).

```text
                    ┌─ Ngày 2: system prompt, few-shot, định dạng JSON ─┐
                    ▼                                                   │
Câu hỏi ──► tokenizer ──► token ──► LLM ──► xác suất token ──► sampling ──► câu trả lời
            (Ngày 1)                         (Ngày 1: temperature, top_p, stop, chi phí)
                                   ▲
                                   │ đoạn văn liên quan được chèn vào prompt  (Tuần 2–5: RAG)
                                   │
Tài liệu ──► đoạn văn ──► embedding model ──► vector ──► vector database ──► top-k đoạn gần nhất
                          (Ngày 4)                       (Ngày 5)

LangChain (Ngày 3) là lớp keo nối các bước trên thành một chuỗi gọi được bằng invoke / stream / batch.
```

---

## 3. Môi trường tham chiếu

Các con số trong bộ tài liệu này được đo thật trên máy của bạn ngày **2026-09-26**, dùng đúng môi trường đã dựng ở [Tuần 0](../ollama_setup.md). Khi bạn đo lại, con số có thể lệch một chút — điều quan trọng là **tỷ lệ và xu hướng**.

| Thành phần | Giá trị | Nguồn |
|---|---|---|
| Ollama | phiên bản 0.34.4 | log khởi động |
| `qwen3:4b` | 4.0 tỷ tham số, lượng tử hóa Q4_K_M, context tối đa 262144 token | `ollama show qwen3:4b` |
| Tham số mặc định của `qwen3:4b` | temperature 0.6, top_k 20, top_p 0.95, repeat_penalty 1 | `ollama show qwen3:4b` |
| `bge-m3` | 566.7 triệu tham số, F16, vector 1024 chiều, context 8192 token | `ollama show bge-m3` |
| Context đang chạy | 8192 token (biến môi trường `OLLAMA_CONTEXT_LENGTH`) | [ollama_setup.md](../ollama_setup.md) |
| Tốc độ sinh token | khoảng 48 token/giây | [progress_tracker.md](../progress_tracker.md) |
| Tốc độ xử lý prompt | khoảng 2280 token/giây (prompt 2558 token) | đo khi viết tài liệu này |

---

## 4. Bảng thuật ngữ

Quy tắc của repo là không viết tắt. Các từ viết tắt tiếng Anh dưới đây là tên chuẩn của ngành nên vẫn được dùng, nhưng mỗi bài đều ghi tên đầy đủ ở lần xuất hiện đầu tiên.

| Thuật ngữ | Tên đầy đủ | Nghĩa ngắn gọn |
|---|---|---|
| LLM | Large Language Model | Mô hình ngôn ngữ lớn, sinh văn bản bằng cách đoán token tiếp theo |
| API | Application Programming Interface | Giao diện lập trình để gọi model từ code |
| Token | — | Mảnh văn bản nhỏ nhất model xử lý (một từ, một phần của từ, hoặc một byte) |
| BPE | Byte Pair Encoding | Thuật toán tạo từ vựng token phổ biến nhất |
| NFC / NFD | Normalization Form Canonical Composition / Decomposition | Hai cách lưu chữ có dấu trong Unicode: dựng sẵn và tổ hợp |
| Context window | — | Tổng số token tối đa model xử lý trong một lần (đầu vào cộng đầu ra) |
| KV cache | Key-Value cache | Bộ nhớ đệm các ma trận khóa và giá trị của attention, giúp không tính lại phần prompt đã xử lý |
| Prefill / Decode | — | Pha xử lý toàn bộ prompt song song / pha sinh từng token một |
| JSON | JavaScript Object Notation | Định dạng dữ liệu có cấu trúc, dùng cho structured output |
| LCEL | LangChain Expression Language | Cách ghép các bước LangChain bằng toán tử pipe (dấu gạch đứng) |
| Embedding | — | Vector số biểu diễn ngữ nghĩa của một đoạn văn |
| MTEB | Massive Text Embedding Benchmark | Bộ đánh giá chuẩn cho embedding model |
| ANN | Approximate Nearest Neighbor | Tìm láng giềng gần nhất xấp xỉ, đổi một ít độ chính xác lấy tốc độ |
| HNSW | Hierarchical Navigable Small World | Chỉ mục ANN dạng đồ thị nhiều tầng, dùng trong Chroma và Qdrant |
| RAG | Retrieval-Augmented Generation | Truy xuất tài liệu liên quan rồi đưa vào prompt để LLM trả lời |
| VRAM | Video Random Access Memory | Bộ nhớ của GPU (Graphics Processing Unit), máy bạn có 6 GB |
| p50 / p95 | percentile 50 / percentile 95 | Phân vị 50 (trung vị) / phân vị 95 của latency |

---

## 5. Nguồn chính của tuần

Danh sách đầy đủ ở [resources.md](../resources.md). Nguyên tắc: **tài liệu chính thức trước, blog sau, paper cuối cùng**.

- Anthropic Claude Docs — prompt engineering, prompt caching, token counting.
- LangChain Python — mục *Runnable interface*, *LCEL*, *Streaming*.
- Sentence Transformers — embedding, cosine similarity, chuẩn hóa.
- MTEB Leaderboard — chọn embedding model, nhớ lọc theo đa ngữ.
- Chroma Docs — collection, query, metadata filter, cấu hình khoảng cách.
