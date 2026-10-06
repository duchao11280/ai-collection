# Ngày 4 — Embeddings

Bài thực hành đi kèm: tự code semantic search bằng numpy trên khoảng 200 đoạn văn ([kế hoạch Tuần 1](../learning_plan.md)).
Model dùng trong bài: `bge-m3` qua Ollama (`POST /api/embed`).

!!! abstract "Sau bài này bạn trả lời được"
    - Embedding model khác LLM ở đâu, được huấn luyện thế nào?
    - Vì sao sau khi chuẩn hóa vector thì cosine, dot product và khoảng cách L2 cho cùng một thứ hạng?
    - Vì sao điểm cosine 0.45 **không** nói lên được kết quả có liên quan hay không?
    - Embedding 1024 chiều cho 100 000 đoạn văn tốn bao nhiêu RAM?

---

## 1. Embedding là gì — bạn đã biết rồi

Nếu từng làm person re-identification (trong repo có model `person-reidentification-retail-0287` ở `code/object_detection/modelvino/`), bạn đã dùng embedding: một mạng CNN (Convolutional Neural Network — mạng nơ-ron tích chập) biến ảnh một người thành vector, hai ảnh của cùng một người cho hai vector gần nhau, so sánh bằng cosine.

Text embedding y hệt, chỉ khác đầu vào: **embedding model** biến một đoạn văn thành vector $v \in \mathbb{R}^d$ sao cho **nghĩa gần nhau ↔ vector gần nhau**. Với `bge-m3`, $d = 1024$.

Phân biệt với LLM: LLM *sinh* chữ; embedding model chỉ *đọc* rồi nén cả đoạn văn thành một vector, không sinh gì cả. Trong RAG (Retrieval-Augmented Generation), embedding model lo phần **tìm**, LLM lo phần **trả lời**.

---

## 2. Embedding model được làm ra thế nào

### 2.1. Kiến trúc bi-encoder

```text
"Hà Nội là thủ đô..." → tokenizer → Transformer encoder → một vector cho mỗi token
                      → pooling (gộp thành một vector) → chuẩn hóa → vector 1024 chiều
```

- **Pooling**: lấy vector của token đặc biệt ở đầu câu (CLS pooling — `bge-m3` dùng cách này), hoặc lấy trung bình vector của mọi token (mean pooling — họ `multilingual-e5` dùng cách này).
- Câu hỏi và tài liệu được mã hóa **độc lập** bởi cùng một model. Vector tài liệu tính **trước một lần** rồi lưu lại; lúc tìm chỉ cần mã hóa câu hỏi. Đây là lý do tìm kiếm nhanh.
- Đối lập là **cross-encoder** (reranker, Tuần 4): đọc câu hỏi và tài liệu **cùng lúc** → chính xác hơn, nhưng không tính trước được, phải chạy lại cho từng cặp (câu hỏi, tài liệu).

### 2.2. Huấn luyện bằng contrastive learning

Giống metric learning bên thị giác máy tính (triplet loss, ArcFace): dữ liệu là các cặp (câu hỏi $q$, đoạn văn đúng $p^+$). Hàm mất mát InfoNCE kéo $q$ lại gần $p^+$ và đẩy xa các đoạn khác trong batch:

$$\mathcal{L} = -\log \frac{\exp\left(s(q, p^+) / \tau\right)}{\sum_{p \in \text{batch}} \exp\left(s(q, p) / \tau\right)}$$

- $s$ là cosine similarity, $\tau$ là temperature, thường rất nhỏ (khoảng 0.01–0.05).
- Các đoạn khác trong cùng batch đóng vai mẫu âm (*in-batch negatives*). Thêm **hard negatives** — những đoạn trông giống nhưng sai — giúp model phân biệt tinh hơn.

Hệ quả quan trọng của cách huấn luyện này: model chỉ học **thứ tự** (đoạn đúng phải có điểm cao hơn đoạn sai), không học một thang điểm tuyệt đối có ý nghĩa. Số đo ở mục 3.3 cho thấy rõ điều này.

---

## 3. Đo độ tương đồng

### 3.1. Ba cách đo

$$\text{dot}(a, b) = \sum_i a_i b_i \qquad \cos(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|} \qquad \text{L2}(a, b) = \|a - b\| = \sqrt{\sum_i (a_i - b_i)^2}$$

- **Cosine** chỉ quan tâm **hướng**, bỏ qua độ dài vector. Giá trị trong khoảng $[-1, 1]$, càng lớn càng giống.
- **Dot product** quan tâm cả hướng lẫn độ dài.
- **L2** là khoảng cách Euclid: **càng nhỏ càng giống** — ngược chiều với hai cách trên.

### 3.2. Chuẩn hóa vector: ba cách đo trở thành một

Chuẩn hóa L2: $\hat{v} = v / \|v\|$, mọi vector có độ dài bằng 1. Khi đó:

$$\cos(\hat{a}, \hat{b}) = \hat{a} \cdot \hat{b}$$

$$\|\hat{a} - \hat{b}\|^2 = \|\hat{a}\|^2 + \|\hat{b}\|^2 - 2\,\hat{a} \cdot \hat{b} = 2 - 2\cos(\hat{a}, \hat{b})$$

→ Với vector đã chuẩn hóa, **cosine, dot product và L2 cho cùng một thứ hạng**. Lợi ích:

- Tính cosine giữa câu hỏi và cả corpus chỉ là **một phép nhân ma trận** — cách nhanh nhất trên numpy và GPU.
- Chọn metric nào trong vector database cũng ra cùng thứ hạng (quan trọng cho Ngày 5).

Kiểm tra trên máy bạn: `/api/embed` của Ollama trả vector `bge-m3` có độ dài 1.0000002 → **đã được chuẩn hóa sẵn**. Nhưng đừng giả định điều này cho mọi nguồn — ví dụ `sentence-transformers` chỉ chuẩn hóa khi truyền `normalize_embeddings=True`. Mỗi lần đổi model hoặc thư viện, tự kiểm tra `numpy.linalg.norm` một lần.

### 3.3. Điểm số thật từ bge-m3

Cosine giữa câu gốc "Thủ đô của Việt Nam là Hà Nội." và các câu khác, đo trên máy bạn:

| Câu so sánh | Cosine | Nhận xét |
|---|---|---|
| Thủ đô của Việt Nam là Hà Nội. | 1.000 | Chính nó |
| Hà Nội là thủ đô nước ta. | 0.902 | Diễn đạt khác, cùng nghĩa → rất cao |
| Hanoi is the capital of Vietnam. | 0.673 | **Khác ngôn ngữ** vẫn khớp → model đa ngữ |
| Giá vàng hôm nay tăng mạnh. | 0.443 | Không liên quan, nhưng điểm không gần 0 |
| Tôi thích ăn phở vào buổi sáng. | 0.386 | Không liên quan |

Bốn bài học:

1. **Câu không liên quan vẫn được khoảng 0.4**, không phải 0. Vector của model thực tế chỉ nằm trong một "hình nón" hẹp của không gian (hiện tượng *anisotropy* — bất đẳng hướng), một phần do temperature $\tau$ rất nhỏ khi huấn luyện.
2. **Điểm tuyệt đối không được hiệu chỉnh**: "giá vàng" có điểm cao hơn "phở" dù cả hai đều không liên quan. Đừng đặt ngưỡng kiểu "cosine > 0.5 là liên quan" theo cảm tính. Nếu cần ngưỡng, hãy hiệu chỉnh trên dữ liệu của bạn, cho **từng** model.
3. **Không so điểm giữa hai model khác nhau**: 0.8 của model này không tương đương 0.8 của model kia. So sánh model bằng thứ hạng (Recall@k ở Tuần 3).
4. **Tìm kiếm chéo ngôn ngữ hoạt động**: câu hỏi tiếng Việt vẫn tìm được tài liệu tiếng Anh, nhưng điểm thấp hơn khi cùng ngôn ngữ. Trong corpus trộn nhiều ngôn ngữ, tài liệu cùng ngôn ngữ với câu hỏi dễ được xếp lên trên.

---

## 4. Semantic search bằng numpy

Đây là tìm kiếm **chính xác** (exact k-nearest-neighbor, còn gọi là vét cạn — brute force):

1. Mã hóa $N$ đoạn văn → ma trận $E \in \mathbb{R}^{N \times d}$, mỗi hàng đã chuẩn hóa.
2. Mã hóa câu hỏi → $q \in \mathbb{R}^{d}$, đã chuẩn hóa.
3. Điểm: $s = E\,q \in \mathbb{R}^{N}$.
4. Lấy $k$ vị trí có điểm cao nhất.

```python
import numpy


def normalize(matrix: numpy.ndarray) -> numpy.ndarray:
    return matrix / numpy.linalg.norm(matrix, axis=-1, keepdims=True)


def search(query_vector: numpy.ndarray, corpus_matrix: numpy.ndarray, top_k: int = 5):
    scores = corpus_matrix @ query_vector                      # (N,)
    candidates = numpy.argpartition(-scores, top_k)[:top_k]    # O(N), chưa sắp xếp; cần top_k < N
    ranked = candidates[numpy.argsort(-scores[candidates])]    # chỉ sắp xếp k phần tử
    return [(int(position), float(scores[position])) for position in ranked]
```

- `argpartition` tìm $k$ phần tử lớn nhất trong $O(N)$ thay vì sắp xếp toàn bộ trong $O(N \log N)$.
- Nhiều câu hỏi cùng lúc: $S = Q\,E^\top$ với $Q \in \mathbb{R}^{m \times d}$ — vẫn chỉ là một phép nhân ma trận.
- Lưu `corpus_matrix` ở kiểu `float32`: numpy mặc định tạo `float64` từ list số Python → tốn gấp đôi RAM mà không lợi gì.

Chi phí mỗi truy vấn là $N \times d$ phép nhân-cộng, và toàn bộ ma trận phải được đọc qua bộ nhớ:

| Số đoạn $N$ | Phép tính mỗi truy vấn ($d = 1024$) | Ma trận `float32` | Thời gian cỡ (CPU) |
|---|---|---|---|
| 200 | khoảng 200 nghìn | 0.8 MB | Micro giây |
| 100 000 | khoảng 100 triệu | 410 MB | Vài chục mili giây |
| 1 000 000 | khoảng 1 tỷ | 4.1 GB | Vài trăm mili giây → cần ANN (Tuần 3) |

Ở quy mô vài trăm đoạn, **thời gian mã hóa câu hỏi thường lâu hơn thời gian tìm kiếm**. Khi đo latency ở mini project cuối tuần, hãy tách riêng hai phần này.

---

## 5. Tìm kiếm bất đối xứng và tiền tố

Câu hỏi thì ngắn ("máy nén khí bị nóng?"), tài liệu thì dài và viết theo văn phong khác. Nhiều model được huấn luyện để xử lý hai loại văn bản này khác nhau:

| Model | Cách dùng đúng |
|---|---|
| `bge-m3` | Không cần tiền tố khi tìm kiếm |
| `multilingual-e5-large` (và bản `base`, `small`) | **Bắt buộc**: câu hỏi thêm `"query: "` ở đầu, tài liệu thêm `"passage: "`. Thiếu tiền tố → chất lượng giảm mà không báo lỗi |
| `multilingual-e5-large-instruct` | Câu hỏi có dạng `"Instruct: {mô tả nhiệm vụ}\nQuery: {câu hỏi}"`, tài liệu để nguyên |
| Model khác | Đọc model card — mỗi model một quy ước |

**Quy tắc vàng: cùng model, cùng cách tiền xử lý** cho cả lúc lập chỉ mục lẫn lúc truy vấn. Đổi model → phải mã hóa lại toàn bộ corpus, vì vector của hai model không so sánh được với nhau (có khi còn khác số chiều). Nên lưu tên model vào metadata của từng bản ghi.

---

## 6. Độ dài tối đa và vì sao phải chia nhỏ văn bản

- Mỗi model có giới hạn token đầu vào: `bge-m3` 8192, `multilingual-e5` 512. Vượt quá thì **phần đuôi bị cắt âm thầm** — thông tin ở cuối văn bản không có mặt trong vector.
- Dù không vượt giới hạn, một vector cho cả văn bản dài là "trung bình" của nhiều ý → từng ý cụ thể bị pha loãng, câu hỏi hẹp khó khớp.
- → Chia văn bản thành các đoạn (chunk) vừa phải. Tuần 1 chỉ cần tách theo đoạn văn; Tuần 2 – Ngày 3 sẽ so sánh 5 chiến lược chunking.

---

## 7. Chọn embedding model cho tiếng Việt

| Model | Số chiều | Token tối đa | Chạy ở đâu | Ghi chú |
|---|---|---|---|---|
| `bge-m3` (BAAI) | 1024 | 8192 | Local: Ollama (1.2 GB, F16), `sentence-transformers` | Đa ngữ tốt, context dài, giấy phép MIT. Model còn sinh được vector thưa (sparse) và đa vector, nhưng qua Ollama chỉ lấy được vector dày (dense) |
| `multilingual-e5-large` (intfloat) | 1024 | 512 | Local: `sentence-transformers` | Cần tiền tố; bản `base` (768 chiều) và `small` (384 chiều) nhẹ hơn; giấy phép MIT |
| `text-embedding-3-small` / `text-embedding-3-large` (OpenAI) | 1536 / 3072, rút gọn được bằng tham số `dimensions` | 8191 | API trả phí | Dữ liệu phải gửi ra ngoài; rút gọn được số chiều nhờ cách huấn luyện kiểu Matryoshka |

Ngoài ra trên Hugging Face có các bản `bge-m3` được tinh chỉnh riêng cho tiếng Việt — đáng thử, nhưng chỉ tin sau khi đo trên dữ liệu của bạn.

Tiêu chí chọn, theo thứ tự ưu tiên:

1. **Chất lượng trên dữ liệu của bạn**, không phải điểm leaderboard. MTEB (Massive Text Embedding Benchmark) hữu ích để lọc danh sách ứng viên (nhớ lọc theo đa ngữ), nhưng không thay được 30–50 câu hỏi vàng của chính bạn (Tuần 3 – Ngày 1).
2. **Dữ liệu có được phép rời khỏi máy không**: tài liệu nội bộ thì ưu tiên chạy local.
3. **Độ dài tối đa** so với kích thước chunk dự định.
4. **Số chiều** → RAM, dung lượng lưu trữ, tốc độ tìm (mục 8).
5. **Tốc độ mã hóa và chi phí**.
6. **Giấy phép sử dụng**.

---

## 8. Số chiều, RAM và dung lượng

Đây là câu hỏi tự kiểm tra của tuần: *embedding 1024 chiều cho 100 000 chunk tốn bao nhiêu RAM?*

$$\text{Bộ nhớ cho vector} = N \times d \times \text{số byte mỗi giá trị}$$

Với $N = 100\,000$ và $d = 1024$:

| Kiểu lưu | Byte mỗi giá trị | Dung lượng |
|---|---|---|
| `float32` (mặc định) | 4 | 409 600 000 byte ≈ **409.6 MB** (≈ 390.6 MiB) |
| `float16` | 2 | ≈ 204.8 MB |
| `int8` (lượng tử hóa) | 1 | ≈ 102.4 MB |
| Nhị phân (1 bit mỗi chiều) | 1/8 | ≈ 12.8 MB |

Đó mới chỉ là vector. Một hệ thống thật còn cần:

- **Chỉ mục HNSW** (Ngày 5, Tuần 3): mỗi vector lưu danh sách hàng xóm. Với `max_neighbors = 16`, tầng dưới cùng tốn khoảng $2 \times 16 \times 4 = 128$ byte mỗi vector → khoảng 12.8 MB, cộng thêm các tầng trên (nhỏ hơn nhiều).
- **Văn bản gốc và metadata**: 100 000 chunk × khoảng 1.5 KB văn bản tiếng Việt (UTF-8) ≈ 150 MB. Phần này thường nằm trên đĩa, không bắt buộc phải nằm trong RAM.
- **Bộ nhớ làm việc** của tiến trình.

→ Trả lời gọn: **khoảng 410 MB cho riêng vector `float32`; thực tế nên dự trù 0.5–0.7 GB**. Có thể giảm bằng `float16`/`int8` hoặc dùng ít chiều hơn (Matryoshka), đổi lại mất một ít chất lượng — và phải đo mới biết mất bao nhiêu.

---

## 9. Chi phí mã hóa

- **Local**: tính bằng thời gian. `/api/embed` trả về `prompt_eval_count` là số token đã mã hóa. Câu 73 ký tự ở Ngày 1 tốn 24 token `bge-m3` (gồm 2 token đặc biệt ở đầu và cuối).
- **API**: tổng token × giá mỗi 1 triệu token. Giá embedding thường rẻ hơn giá LLM rất nhiều → khoản tốn kém thật sự là **mã hóa lại toàn bộ khi đổi model**, không phải lần mã hóa đầu tiên.
- **Gửi theo lô** (tham số `input` là danh sách nhiều đoạn) nhanh hơn nhiều so với gửi từng đoạn: giảm chi phí gọi mạng và tận dụng được tính toán song song của GPU.
- Khi vừa chạy `qwen3:4b` vừa mã hóa hàng loạt, hai model tranh nhau 6 GB VRAM và có thể thay nhau nạp/gỡ khỏi bộ nhớ. Cách xử lý có trong mục xử lý sự cố của [ollama_setup.md](../ollama_setup.md).
- Chuẩn hóa NFC trước khi mã hóa. `bge-m3` tình cờ miễn nhiễm với NFD (Ngày 1, mục 2.4), nhưng model khác thì không.

```python
import httpx
import numpy

response = httpx.post(
    "http://localhost:11434/api/embed",
    json={"model": "bge-m3", "input": paragraphs[:32]},   # một lô 32 đoạn
    timeout=300,
)
data = response.json()
batch_matrix = numpy.array(data["embeddings"], dtype=numpy.float32)   # (32, 1024)
token_count = data["prompt_eval_count"]                                 # tổng token của cả lô
```

---

## 10. Lỗi thường gặp

| Lỗi | Hậu quả |
|---|---|
| Lập chỉ mục bằng model A, truy vấn bằng model B | Kết quả vô nghĩa, hoặc lỗi lệch số chiều |
| Quên tiền tố `query: ` / `passage: ` của họ e5 | Chất lượng giảm mà không có lỗi nào báo |
| Dùng dot product với vector chưa chuẩn hóa | Vector có độ lớn lớn được ưu tiên sai |
| Mã hóa cả tài liệu dài thành một vector | Ý bị pha loãng, phần đuôi bị cắt |
| Đặt ngưỡng điểm theo cảm tính | Lọc nhầm kết quả đúng hoặc giữ lại kết quả rác |
| Để numpy dùng `float64` | Tốn gấp đôi RAM |

---

## 11. Câu hỏi tự kiểm tra

1. Chứng minh $\|\hat{a} - \hat{b}\|^2 = 2 - 2\cos(\hat{a}, \hat{b})$ với hai vector đơn vị.
2. Vì sao bi-encoder nhanh hơn cross-encoder rất nhiều khi tìm trong 1 triệu đoạn văn?
3. Đổi từ `bge-m3` sang `multilingual-e5-large` (cùng 1024 chiều). Có dùng lại vector cũ được không?
4. Top-1 của một câu hỏi có cosine 0.45. Kết quả đó có liên quan không?
5. 500 000 chunk, 768 chiều, lưu `float16`: bao nhiêu MB cho riêng vector?

### Gợi ý đáp án

1. Khai triển $\|\hat{a} - \hat{b}\|^2 = \hat{a}\cdot\hat{a} - 2\,\hat{a}\cdot\hat{b} + \hat{b}\cdot\hat{b} = 1 - 2\cos + 1$.
2. Bi-encoder mã hóa 1 triệu đoạn **một lần** từ trước; lúc tìm chỉ mã hóa câu hỏi rồi làm phép nhân ma trận (hoặc tra chỉ mục ANN). Cross-encoder phải chạy model 1 triệu lần cho mỗi câu hỏi.
3. Không. Hai model học hai không gian vector khác nhau; trùng số chiều chỉ khiến lỗi **không bị phát hiện** thay vì báo lỗi. Phải mã hóa lại toàn bộ.
4. Không kết luận được từ con số đó. Với `bge-m3`, câu hoàn toàn không liên quan cũng đạt khoảng 0.4. Phải so với điểm của các kết quả khác và với ngưỡng đã hiệu chỉnh trên dữ liệu của bạn.
5. $500\,000 \times 768 \times 2 = 768\,000\,000$ byte ≈ 768 MB.

---

## Đọc thêm

- Sentence Transformers Docs — *Semantic Search*, *Computing Embeddings* (mục chuẩn hóa).
- MTEB Leaderboard trên Hugging Face — lọc theo đa ngữ.
- Model card của `BAAI/bge-m3` và `intfloat/multilingual-e5-large` — cách dùng tiền tố, độ dài tối đa.
