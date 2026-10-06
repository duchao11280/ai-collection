# Cuối tuần 1 — Mini project semantic search và ôn tập

Bài thực hành: semantic search trên tài liệu tiếng Việt thật của bạn (300–500 đoạn). Đo thời gian lập chỉ mục, latency truy vấn và chi phí embedding ([kế hoạch Tuần 1](../learning_plan.md)).
Buổi cuối tuần kéo dài 2–3 giờ: phần lý thuyết dưới đây tập trung vào **cách đo cho đúng** — thứ sẽ theo bạn suốt 15 tuần còn lại.

---

## 1. Kiến trúc mini project

Gợi ý nguồn dữ liệu nếu chưa có sẵn: chính các ghi chú tiếng Việt trong `docs/` của repo này (thư mục `machine_learning/`, `LLM/`), hoặc tài liệu kỹ thuật ở chỗ làm.

```text
Lập chỉ mục (chạy bao nhiêu lần cũng không nhân đôi dữ liệu):
  file .md / .txt ─► đọc ─► chuẩn hóa NFC ─► tách đoạn theo dòng trống ─► bỏ đoạn quá ngắn
                  ─► mã hóa theo lô bằng bge-m3 ─► upsert vào Chroma (ID tất định + metadata)

Truy vấn:
  câu hỏi ─► chuẩn hóa NFC ─► mã hóa bằng bge-m3 ─► Chroma query top-5 ─► in điểm, nguồn, trích đoạn
```

Kiến thức dùng lại từ các ngày:

| Bước | Lấy từ |
|---|---|
| Chuẩn hóa NFC, đếm token | Ngày 1, mục 2.4 và 6.4 |
| Mã hóa theo lô, kiểm tra vector đã chuẩn hóa | Ngày 4, mục 3.2 và 9 |
| ID tất định, `upsert`, metadata, đổi khoảng cách ra cosine | Ngày 5, mục 4, 6 và 7 |
| Chain LangChain (không bắt buộc) | Ngày 3 — dùng nếu muốn luyện thêm |

---

## 2. Đo đạc cho đúng

### 2.1. Thời gian lập chỉ mục

Đo **theo từng giai đoạn** — một con số tổng không cho biết cần tối ưu chỗ nào:

| Giai đoạn | Đo gì | Dự đoán |
|---|---|---|
| Đọc file và tách đoạn | Giây | Không đáng kể |
| Mã hóa | Giây, **đoạn/giây**, **token/giây** | Chiếm phần lớn thời gian |
| Upsert vào Chroma | Giây | Nhỏ ở quy mô này |

Thử hai kích thước lô (ví dụ 1 đoạn và 32 đoạn mỗi lần gọi `/api/embed`) để thấy tận mắt lợi ích của mã hóa theo lô.

### 2.2. Latency truy vấn

- **Khởi động trước (warm-up).** Lần gọi đầu tiên phải nạp model vào VRAM, `load_duration` có thể tới vài giây. Bỏ vài lần gọi đầu khỏi thống kê, nhưng **ghi riêng** con số "khởi động lạnh" vì người dùng thật sẽ gặp nó (Ollama gỡ model sau 5 phút không dùng).
- **Tách hai phần**: thời gian mã hóa câu hỏi và thời gian tìm trong Chroma. Ở vài trăm đoạn, mã hóa thường chiếm phần lớn.
- **Báo cáo phân vị, không báo trung bình.** Chạy ít nhất 30–50 câu hỏi, báo p50 (phân vị 50, tức trung vị) và p95 (phân vị 95: 95% truy vấn nhanh hơn con số này). Trung bình bị kéo lệch bởi vài lần chậm bất thường và che mất trải nghiệm tệ nhất. Lưu ý: với 30 mẫu, p95 gần như chỉ là lần chậm thứ hai — càng nhiều mẫu, con số càng đáng tin.
- Dùng `time.perf_counter()` (đồng hồ độ phân giải cao, đơn điệu), không dùng `time.time()`.

```python
import time

import numpy

embedding_latencies = []
search_latencies = []
for question in questions:
    started = time.perf_counter()
    query_vector = embed_one(question)                      # gọi /api/embed
    embedded = time.perf_counter()
    collection.query(query_embeddings=[query_vector], n_results=5)
    finished = time.perf_counter()
    embedding_latencies.append((embedded - started) * 1000)
    search_latencies.append((finished - embedded) * 1000)

for name, values in [("mã hóa", embedding_latencies), ("tìm kiếm", search_latencies)]:
    print(name, "p50", numpy.percentile(values, 50), "p95", numpy.percentile(values, 95), "mili giây")
```

### 2.3. Chi phí embedding

- Tổng token = cộng `prompt_eval_count` của mọi lần gọi `/api/embed`.
- **Local**: chi phí là thời gian mã hóa (và điện năng).
- **Quy đổi giả định sang API** để có cảm giác về độ lớn: 500 đoạn × khoảng 100 token = 50 000 token; với giá giả định 0.02 USD cho mỗi 1 triệu token → 0.001 USD.
- Kết luận thường gặp: **mã hóa rẻ, sinh chữ đắt**. Chi phí của một hệ thống RAG nằm chủ yếu ở LLM, không phải ở embedding — trừ khi bạn phải mã hóa lại toàn bộ corpus.

### 2.4. Ghi lại điều kiện đo

Một con số chỉ có ý nghĩa khi đi kèm điều kiện đo. Ghi lại: model và mức lượng tử hóa, `num_ctx`, máy có cắm sạc không, `qwen3:4b` có đang chiếm VRAM cùng lúc không, số đoạn, độ dài trung bình mỗi đoạn (ký tự và token), kích thước lô.

---

## 3. Kiểm tra chất lượng nhanh — tiền đề cho Tuần 3

Viết 10 câu hỏi mà bạn **biết trước đoạn trả lời đúng**, đếm số câu có đoạn đúng nằm trong top-5. Đó là Recall@5 (tỷ lệ câu hỏi tìm thấy đáp án trong 5 kết quả đầu) phiên bản thủ công. Nên có đủ các loại:

| Loại câu hỏi | Mục đích |
|---|---|
| Dùng đúng từ ngữ trong tài liệu | Mức dễ, phải đúng gần hết |
| Diễn đạt khác hẳn, không trùng từ khóa | Chỗ semantic search hơn hẳn tìm từ khóa |
| Hỏi bằng tiếng Anh cho tài liệu tiếng Việt | Kiểm tra khả năng chéo ngôn ngữ của `bge-m3` |
| Không có đáp án trong tài liệu | Xem điểm top-1 thấp tới đâu — nhớ bài học "điểm không được hiệu chỉnh" ở Ngày 4 |
| Chứa tên riêng, mã số, ký hiệu chuyên ngành | Semantic search thường yếu ở đây; Tuần 3 dùng BM25 và hybrid search để bù |

Ghi kết quả vào note. Tuần 3 sẽ biến danh sách này thành bộ câu hỏi vàng 30–50 câu và script benchmark tái lập được.

---

## 4. Đáp án câu hỏi tự kiểm tra của tuần

### 4.1. Vì sao cùng một câu hỏi lại tốn số token khác nhau giữa 2 model?

1. **Tokenizer khác nhau.** Mỗi họ model học từ vựng riêng trên dữ liệu riêng. Cùng câu tiếng Việt 73 ký tự: `qwen3:4b` tốn khoảng 22 token nội dung, trong khi câu tiếng Anh tương đương chỉ tốn 17 token.
2. **Chat template khác nhau.** `qwen3:4b` tốn khoảng 10 token khung cho mỗi request; model khác có khung dài ngắn khác.
3. **Phần ẩn khác nhau**: system prompt mặc định, định nghĩa tool, cách mã hóa ảnh.
4. **Token thinking**: model có chế độ suy nghĩ sinh thêm token đầu ra.
5. Bonus từ số đo: **cùng một model** cũng có thể tốn gấp 3 lần nếu văn bản ở dạng Unicode NFD (92 so với 32 token).

### 4.2. Viết được một LCEL chain 3 bước mà không cần copy-paste

Tự kiểm tra bằng cách viết lại từ đầu chain ở Ngày 3, mục 3.1, rồi trả lời được:

- [ ] Kiểu dữ liệu ở mỗi mũi tên của `prompt | model | StrOutputParser()`.
- [ ] Vì sao cần `RunnablePassthrough.assign` để bước 3 dùng được kết quả của cả bước 1 lẫn bước 2.
- [ ] Bước nào trong chain thật sự stream được, và vì sao.
- [ ] Nếu prompt chứa ví dụ JSON thì phải viết dấu ngoặc nhọn thế nào.

### 4.3. Embedding 1024 chiều cho 100 000 chunk tốn bao nhiêu RAM?

$$100\,000 \times 1024 \times 4 \text{ byte} = 409\,600\,000 \text{ byte} \approx 409.6 \text{ MB} \ (\approx 390.6 \text{ MiB})$$

- Đó là riêng vector `float32`. Cộng chỉ mục HNSW (khoảng 13 MB với `max_neighbors = 16`), metadata, văn bản (thường nằm trên đĩa) và bộ nhớ làm việc → **thực tế dự trù 0.5–0.7 GB**.
- `float16` còn khoảng 205 MB, `int8` khoảng 102 MB — đổi lại mất một ít chất lượng, phải đo.
- Chi tiết: [Ngày 4, mục 8](day_4_embeddings.md).

---

## 5. Ôn tập toàn tuần

1. Prefill và decode là gì? Pha nào quyết định thời gian của một câu trả lời dài?
2. Văn bản tiếng Việt copy từ PDF bị tách dấu (NFD). Theo số đo, nó ảnh hưởng thế nào tới `qwen3:4b` và `bge-m3`?
3. Muốn câu trả lời ổn định hơn, nên chỉnh temperature hay top_p trước? Có nên chỉnh cả hai cùng lúc không?
4. Trong bốn mức đảm bảo structured output, mức nào cần validate bằng Pydantic?
5. Vì sao đặt thời gian hiện tại ở đầu system prompt làm tăng chi phí và latency?
6. `prompt | model` (không có parser) trả về kiểu gì?
7. Vì sao `batch` 20 request tới Ollama trên máy bạn không nhanh gấp 20 lần?
8. Viết công thức liên hệ khoảng cách L2 bình phương và cosine của hai vector đã chuẩn hóa.
9. Vì sao không được so điểm cosine của `bge-m3` với điểm của `multilingual-e5`?
10. Chroma mặc định dùng hàm embedding nào, và vì sao nó nguy hiểm với tiếng Việt?

### Gợi ý đáp án

1. Prefill xử lý cả prompt song song (khoảng 2280 token/giây trên máy bạn); decode sinh từng token một (khoảng 48 token/giây). Câu trả lời dài bị chi phối bởi decode.
2. `qwen3:4b` tốn gần gấp 3 token (92 so với 32) và hiểu kém hơn; `bge-m3` không bị ảnh hưởng vì tokenizer tự chuẩn hóa. Chuẩn hóa NFC ngay từ đầu pipeline.
3. Chỉnh temperature trước (giảm xuống). Mỗi lần chỉ chỉnh một tham số để biết thay đổi nào gây ra hiệu ứng nào.
4. Tất cả các mức. Constrained decoding chỉ đảm bảo cấu trúc, không đảm bảo nội dung, không chống được JSON bị cắt cụt, và không phải ràng buộc nào của schema cũng được thực thi.
5. Tiền tố thay đổi ở mỗi request nên không bao giờ trúng prompt caching; mọi request phải prefill lại toàn bộ và trả giá đầy đủ.
6. `AIMessage`.
7. Ollama chỉ xử lý đồng thời `OLLAMA_NUM_PARALLEL` request cho mỗi model, và các request song song chia nhau cùng một GPU 6 GB.
8. $\|\hat{a} - \hat{b}\|^2 = 2 - 2\cos(\hat{a}, \hat{b})$.
9. Mỗi model có không gian vector và phân bố điểm riêng; chỉ thứ hạng trong cùng một model mới có ý nghĩa.
10. `all-MiniLM-L6-v2`, vector 384 chiều, huấn luyện chủ yếu trên tiếng Anh. Nếu quên truyền `embeddings`, Chroma âm thầm dùng model này và kết quả tìm kiếm tiếng Việt kém mà không có lỗi nào báo.

---

## 6. Ghi kết quả tuần

Điền vào phần *Nhật ký tuần* trong [progress_tracker.md](../progress_tracker.md), gồm:

- Số buổi hoàn thành trên 6.
- Output: đường dẫn tới script semantic search trong `llm-lab/`.
- Số liệu: số đoạn, số token, thời gian lập chỉ mục (từng giai đoạn), latency p50/p95 (tách mã hóa và tìm kiếm), Recall@5 thủ công trên 10 câu hỏi.
- Điều chỉnh cho tuần sau: những loại câu hỏi mà semantic search làm kém — đó chính là đầu vào cho Tuần 2 (chunking) và Tuần 3 (hybrid search).
