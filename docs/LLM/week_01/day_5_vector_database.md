# Ngày 5 — Vector database 101 với Chroma

Bài thực hành đi kèm: dùng Chroma local để upsert, query, lọc metadata, lưu bền vững; so sánh với kết quả numpy hôm trước — phải trùng nhau ([kế hoạch Tuần 1](../learning_plan.md)).
Cài đặt: `pip install chromadb` trong môi trường conda `llm`.

!!! abstract "Sau bài này bạn trả lời được"
    - Đã có numpy thì cần vector database để làm gì?
    - Tìm chính xác và tìm xấp xỉ (ANN) khác nhau thế nào, vì sao ở vài trăm vector chúng cho cùng kết quả?
    - Chroma trả về "khoảng cách" chứ không phải "độ tương đồng" — đổi qua lại thế nào?
    - Làm sao để chạy script nạp dữ liệu hai lần mà không bị nhân đôi?

---

## 1. Đã có numpy, cần vector database để làm gì

| Nhu cầu | numpy (Ngày 4) | Vector database |
|---|---|---|
| Còn dữ liệu sau khi tắt chương trình | Tự `numpy.save` và tự quản lý file văn bản đi kèm | Có sẵn |
| Thêm, sửa, xóa từng bản ghi theo định danh (ID) | Tự viết | `upsert`, `delete` |
| Lọc theo metadata (nguồn, ngày, quyền truy cập) | Tự viết mặt nạ boolean | Tham số `where` |
| Tìm nhanh trong hàng triệu vector | Vét cạn, chi phí $O(N \cdot d)$ | Chỉ mục xấp xỉ (ANN) |
| Nhiều tiến trình, nhiều người dùng cùng truy cập | Không hỗ trợ | Chế độ server |

Với 200–500 đoạn văn của tuần này, numpy **hoàn toàn đủ nhanh**. Mục tiêu của Ngày 5 là học **mô hình thao tác** của một vector database — thứ bạn sẽ dùng lại gần như nguyên vẹn với Qdrant ở Tuần 3.

---

## 2. Tìm chính xác và tìm xấp xỉ

- **Exact search** (tìm chính xác): so câu hỏi với mọi vector. Kết quả đúng tuyệt đối, chi phí tăng tuyến tính theo $N$.
- **ANN** (Approximate Nearest Neighbor — tìm láng giềng gần nhất xấp xỉ): chấp nhận **đôi khi bỏ sót** để nhanh hơn hàng trăm lần. Độ đo chất lượng của ANN là *recall*: tỷ lệ láng giềng thật được tìm thấy.

HNSW (Hierarchical Navigable Small World — đồ thị "thế giới nhỏ" phân tầng) là chỉ mục Chroma dùng. Hình dung:

- Mỗi vector là một đỉnh, nối với vài "hàng xóm" gần nó.
- Có nhiều tầng: tầng trên thưa như đường cao tốc (nhảy xa), tầng dưới dày như đường nội khu (đi chính xác).
- Khi tìm: bắt đầu ở tầng trên cùng, đi tham lam về phía đỉnh gần câu hỏi hơn, hết đường thì xuống tầng dưới và lặp lại.

```text
Tầng 2:  A ─────────────────────── F                  ← vài đỉnh, cạnh dài
Tầng 1:  A ──────── C ──────────── F ──────── H
Tầng 0:  A ── B ── C ── D ── E ── F ── G ── H ── I     ← mọi đỉnh, cạnh ngắn
```

Tham số và giá trị mặc định của Chroma:

| Tham số | Mặc định | Ý nghĩa | Tăng lên thì |
|---|---|---|---|
| `max_neighbors` | 16 | Số hàng xóm tối đa của mỗi đỉnh | Recall tăng, tốn RAM hơn |
| `ef_construction` | 100 | Độ rộng tìm kiếm khi xây chỉ mục | Chỉ mục tốt hơn, xây chậm hơn |
| `ef_search` | 100 | Độ rộng tìm kiếm khi truy vấn | Recall tăng, truy vấn chậm hơn |

Với vài trăm vector, `ef_search = 100` gần như đã duyệt khắp đồ thị → kết quả **trùng với tìm chính xác**. Đó là lý do bài thực hành yêu cầu Chroma phải khớp numpy. Tuần 3 – Ngày 2 sẽ vẽ đường cong recall theo latency khi thay đổi `ef_search` trên dữ liệu lớn hơn.

---

## 3. Mô hình dữ liệu của Chroma

```text
Client ──► Collection (tên, cấu hình khoảng cách và HNSW, hàm embedding)
               └── Bản ghi: id │ embedding │ document │ metadata
```

| Loại client | Khởi tạo | Dùng khi |
|---|---|---|
| Trong bộ nhớ | `chromadb.EphemeralClient()` | Thử nghiệm, unit test — tắt chương trình là mất |
| Lưu đĩa, chạy chung tiến trình | `chromadb.PersistentClient(path="data/chroma")` | Tuần này |
| Client–server | Chạy `chroma run --path data/chroma`, rồi `chromadb.HttpClient(host="localhost", port=8000)` | Nhiều tiến trình, service thật |

Một bản ghi gồm:

- `id`: chuỗi, duy nhất trong collection — khóa để upsert và xóa.
- `embedding`: vector. Mọi bản ghi trong một collection phải **cùng số chiều**, được cố định từ lần thêm dữ liệu đầu tiên.
- `document`: văn bản gốc. Không bắt buộc, nhưng nên có để hiển thị kết quả.
- `metadata`: `dict` phẳng, giá trị kiểu `str`, `int`, `float` hoặc `bool`.

---

## 4. Khoảng cách — chỗ dễ sai nhất

Chroma trả về **khoảng cách** (càng nhỏ càng giống), không phải độ tương đồng:

| `space` | Công thức khoảng cách $d$ | Đổi ra cosine khi vector đã chuẩn hóa |
|---|---|---|
| `l2` (**mặc định**) | $\sum_i (a_i - b_i)^2$ — bình phương, không lấy căn | $\cos = 1 - d / 2$ |
| `cosine` | $1 - \cos(a, b)$ | $\cos = 1 - d$ |
| `ip` (inner product — tích vô hướng) | $1 - a \cdot b$ | $\cos = 1 - d$ |

Cột cuối suy ra trực tiếp từ công thức $\|\hat{a} - \hat{b}\|^2 = 2 - 2\cos$ ở Ngày 4. Với vector đã chuẩn hóa, cả ba `space` cho **cùng thứ hạng**, chỉ khác con số.

`space` được đặt lúc tạo collection và **không đổi được sau đó** — muốn đổi phải tạo collection mới:

```python
import chromadb

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(
    name="week_1_documents",
    configuration={"hnsw": {"space": "cosine"}},
)
```

---

## 5. Cái bẫy của hàm embedding mặc định

Nếu bạn gọi `collection.add(documents=...)` mà **không** truyền `embeddings`, hoặc truy vấn bằng `query_texts=...`, Chroma tự mã hóa văn bản bằng **hàm embedding mặc định**: model `all-MiniLM-L6-v2` chạy local, vector 384 chiều, huấn luyện chủ yếu trên tiếng Anh. Hậu quả:

- Collection đang chứa vector `bge-m3` 1024 chiều → **báo lỗi lệch số chiều** (may mắn, vì còn phát hiện được).
- Collection mới tạo → mọi thứ chạy "bình thường" nhưng kết quả tìm kiếm tiếng Việt **tệ một cách âm thầm**.

→ Tuần này: **tự tính vector bằng `bge-m3`** (dùng lại đúng hàm của Ngày 4), truyền vào bằng `embeddings=` khi ghi và `query_embeddings=` khi truy vấn. Vừa tránh bẫy, vừa đảm bảo phép so sánh với numpy là so trên **cùng một bộ vector**.

---

## 6. Các thao tác

| Thao tác | Hành vi | Ghi chú |
|---|---|---|
| `add` | Thêm mới | ID đã tồn tại thì không ghi đè (bỏ qua hoặc báo lỗi tùy phiên bản) |
| `upsert` | Chưa có thì thêm, có rồi thì ghi đè | **Dùng cái này** cho pipeline có thể chạy lại |
| `update` | Chỉ sửa bản ghi đã tồn tại | |
| `delete` | Xóa theo `ids` hoặc theo điều kiện `where` | |
| `get` | Lấy theo `ids` hoặc `where`, **không xếp hạng** | Kiểm tra dữ liệu |
| `query` | Tìm k bản ghi gần nhất theo vector, có thể kèm bộ lọc | Tìm kiếm |
| `count` | Đếm số bản ghi | Kiểm tra sau khi nạp |

```python
collection.upsert(
    ids=ids,                    # list[str]
    embeddings=vectors,         # numpy array (N, 1024) hoặc list[list[float]]
    documents=paragraphs,       # list[str]
    metadatas=metadatas,        # list[dict]
)

results = collection.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={"source": "operation_manual.md"},
    include=["documents", "metadatas", "distances"],
)
# Kết quả là danh sách lồng nhau: mỗi câu hỏi ứng với một danh sách con
top_documents = results["documents"][0]
top_distances = results["distances"][0]
```

Khi nạp nhiều dữ liệu, chia thành từng lô vài trăm tới vài nghìn bản ghi — Chroma giới hạn kích thước mỗi lần ghi.

### 6.1. Thiết kế ID — nền móng của tính idempotent

*Idempotent* (lũy đẳng): chạy pipeline nạp dữ liệu hai lần cho kết quả giống hệt chạy một lần. Điều kiện: **ID phải tất định** (cùng dữ liệu → cùng ID) **và** dùng `upsert`.

| Cách đặt ID | Ưu điểm | Nhược điểm |
|---|---|---|
| UUID ngẫu nhiên | Không bao giờ trùng | Chạy lại là **nhân đôi dữ liệu** |
| `"{tên file}:{số thứ tự đoạn}"` | Dễ đọc, dễ debug | Chèn một đoạn vào giữa file → mọi ID phía sau trỏ sang nội dung khác |
| Hash nội dung, ví dụ `sha256(tên file + văn bản)` | Nội dung không đổi → ID không đổi; phát hiện được thay đổi | Sửa một chữ → sinh ID mới, phải tự xóa bản ghi cũ |

Tuần này dùng cách 2 hoặc 3 đều được. Tuần 2 – Ngày 5 sẽ dùng hash nội dung để nạp tăng dần (incremental) và dọn bản ghi cũ.

---

## 7. Lọc theo metadata

Các toán tử của `where`:

| Toán tử | Nghĩa | Ví dụ |
|---|---|---|
| `$eq`, `$ne` | Bằng, khác | `{"source": {"$eq": "a.md"}}`, hoặc viết gọn `{"source": "a.md"}` |
| `$gt`, `$gte`, `$lt`, `$lte` | So sánh số | `{"created_at": {"$gte": 1790380800}}` |
| `$in`, `$nin` | Thuộc / không thuộc danh sách | `{"department": {"$in": ["kỹ thuật", "chất lượng"]}}` |
| `$and`, `$or` | Kết hợp nhiều điều kiện | `{"$and": [{"source": "a.md"}, {"paragraph_index": {"$lt": 10}}]}` |

Lọc theo nội dung văn bản dùng tham số riêng: `where_document={"$contains": "bảo trì"}` (và `$not_contains`).

### 7.1. Lọc trước hay lọc sau — lý thuyết cần nắm sớm

- **Lọc sau** (post-filter): tìm top-k trước rồi mới lọc → có thể còn ít hơn k kết quả, thậm chí 0, dù trong database vẫn có đoạn phù hợp.
- **Lọc trước** (pre-filter): chỉ tìm trong tập thỏa điều kiện → đủ k kết quả nếu tồn tại, nhưng khó làm hiệu quả với chỉ mục đồ thị vì bỏ bớt đỉnh làm đồ thị bị "đứt".
- Các vector database thật đều có chiến lược kết hợp. Việc của bạn là **kiểm chứng hành vi**: truy vấn có bộ lọc có trả đủ k kết quả không?

Đây là nền cho lọc theo quyền truy cập — ACL (Access Control List — danh sách kiểm soát truy cập) — ở Tuần 14. Rò rỉ dữ liệu trong RAG thường xảy ra khi quyền được kiểm tra **sau** khi tài liệu đã được đưa vào context.

### 7.2. Metadata cho tuần này

```python
{
    "source": "operation_manual.md",
    "paragraph_index": 12,
    "char_count": 845,
    "embedding_model": "bge-m3",
    "created_at": 1790380800,   # 2026-09-26, lưu dạng số nguyên (Unix timestamp)
}
```

- `created_at` là **số nguyên** vì các toán tử `$gt`, `$lt` chỉ so sánh số; chuỗi `"2026-09-26"` không lọc theo khoảng được.
- Giữ metadata phẳng, cùng kiểu dữ liệu cho cùng một khóa ở mọi bản ghi.
- `embedding_model` giúp phát hiện dữ liệu cũ khi đổi model (Ngày 4, mục 5).
- Tuần 2 – Ngày 4 sẽ mở rộng thành schema đầy đủ có `tenant_id` và `acl_groups`.

---

## 8. Lưu trữ bền vững (persistence)

- `PersistentClient(path=...)` ghi vào một thư mục gồm file SQLite (metadata, văn bản, nhật ký ghi) và các file nhị phân của chỉ mục HNSW. Tắt chương trình rồi mở lại → dữ liệu còn nguyên.
- Cách kiểm tra: nạp dữ liệu, thoát Python, mở một tiến trình mới, `collection.count()` phải ra đúng số cũ.
- Chế độ chạy chung tiến trình **không dành cho nhiều tiến trình cùng ghi** vào một thư mục. Khi có API service (Tuần 11), chuyển sang chế độ client–server.
- Thêm thư mục dữ liệu vào `.gitignore` — thứ gì sinh lại được thì không commit.
- Sao lưu: dừng tiến trình rồi copy nguyên thư mục.

---

## 9. So sánh với numpy — vì sao phải trùng, khi nào không trùng

Cùng một câu hỏi, top-k của Chroma phải trùng top-k numpy ở Ngày 4. Nếu lệch, kiểm tra theo thứ tự:

1. **Có đúng cùng bộ vector không?** Lỡ để Chroma tự mã hóa bằng model mặc định là nguyên nhân phổ biến nhất (mục 5).
2. **Metric**: collection mặc định dùng `l2`. Vector đã chuẩn hóa thì thứ hạng vẫn trùng; vector chưa chuẩn hóa thì thứ hạng L2 khác thứ hạng cosine.
3. **Đổi khoảng cách ra điểm trước khi so**: với `space = "cosine"`, $\cos = 1 - d$. Chênh lệch so với điểm numpy nên nhỏ hơn khoảng $10^{-5}$ (sai số của `float32`).
4. **Điểm bằng nhau** (các đoạn văn trùng lặp): thứ tự giữa chúng có thể khác — không phải lỗi.
5. **Xấp xỉ của HNSW**: ở vài trăm vector gần như không xảy ra; nếu nghi ngờ, tăng `ef_search` rồi thử lại.

```python
numpy_top = search(query_vector, corpus_matrix, top_k=5)                  # hàm ở Ngày 4
numpy_ids = [ids[position] for position, _ in numpy_top]

chroma_result = collection.query(query_embeddings=[query_vector], n_results=5)
chroma_ids = chroma_result["ids"][0]
chroma_scores = [1 - distance for distance in chroma_result["distances"][0]]

assert numpy_ids == chroma_ids, (numpy_ids, chroma_ids)
```

---

## 10. Chroma trong LangChain

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

vector_store = Chroma(
    collection_name="week_1_langchain",
    embedding_function=OllamaEmbeddings(model="bge-m3"),
    persist_directory="data/chroma",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
documents = retriever.invoke("Quy trình bảo trì máy nén khí?")   # retriever là một Runnable
```

- Ở đây LangChain tự gọi `bge-m3` để mã hóa, nên không dính bẫy model mặc định của Chroma.
- Dùng tên collection riêng, đừng dùng chung với collection tạo bằng `chromadb` trực tiếp, và kiểm tra metric của collection mà wrapper tạo ra.
- Retriever là `Runnable` → nối thẳng vào LCEL của Ngày 3:

```python
# prompt, model, RunnablePassthrough, StrOutputParser: như ở Ngày 3
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
```

Đó chính là hình dạng của một chain RAG — công việc của Tuần 2–5 (sẽ thêm bước định dạng danh sách `Document` thành văn bản có trích nguồn trước khi đưa vào prompt).

---

## 11. Bản đồ các vector database

| Công cụ | Loại | Khi nào dùng | Trong lộ trình |
|---|---|---|---|
| Chroma | Chạy chung tiến trình hoặc server, rất đơn giản | Học, prototype, dữ liệu nhỏ | Tuần 1 |
| Qdrant | Server (Docker), lọc payload mạnh, hybrid search | Service thật | Tuần 3 trở đi |
| pgvector | Extension của PostgreSQL | Đã có Postgres, muốn gom về một hệ thống | Tuần 3 (so sánh) |
| Milvus, Weaviate | Server phân tán | Quy mô rất lớn | Tham khảo |
| FAISS | **Thư viện** chỉ mục, không phải database | Nghiên cứu, benchmark chỉ mục | Tuần 3 – Ngày 2 |

---

## 12. Câu hỏi tự kiểm tra

1. Collection dùng `space` mặc định, vector đã chuẩn hóa, Chroma trả khoảng cách 0.2. Cosine là bao nhiêu?
2. Chạy script nạp dữ liệu hai lần, `count()` tăng gấp đôi. Nguyên nhân có thể là gì?
3. Vì sao với 300 đoạn văn, HNSW cho kết quả trùng với tìm chính xác?
4. Lọc sau (post-filter) nguy hiểm thế nào với dữ liệu có phân quyền?
5. Vì sao `created_at` nên lưu dạng số nguyên thay vì chuỗi `"2026-09-26"`?

### Gợi ý đáp án

1. `space` mặc định là `l2` (bình phương): $\cos = 1 - 0.2 / 2 = 0.9$.
2. ID không tất định: sinh ngẫu nhiên (UUID), hoặc phụ thuộc vào thứ tự xử lý thay đổi giữa các lần chạy (ví dụ một bộ đếm tăng dần khi duyệt thư mục theo thứ tự không cố định). Mỗi lần chạy sinh ra ID mới nên `upsert` hay `add` đều coi là bản ghi mới. Sửa: ID tất định từ nội dung hoặc vị trí ổn định, kết hợp `upsert`.
3. `ef_search = 100` và `max_neighbors = 16` đủ để thuật toán duyệt gần hết một đồ thị vài trăm đỉnh, nên hầu như không bỏ sót láng giềng thật nào.
4. Hai rủi ro: (a) sau khi lọc có thể còn quá ít kết quả, người dùng nhận câu trả lời kém mà không ai biết vì sao; (b) nếu việc lọc quyền được làm ở code ứng dụng sau khi đã lấy top-k, tài liệu người dùng không được phép xem vẫn đi qua bộ nhớ, log, cache — chỉ cần một bước quên lọc là rò rỉ vào context. Quyền phải được áp **ngay trong** truy vấn.
5. Các toán tử so sánh `$gt`, `$lt` của Chroma chỉ hoạt động với số; số nguyên cũng không bị mơ hồ về định dạng và múi giờ.

---

## Đọc thêm

- Chroma Docs — *Collections* (cấu hình `space` và HNSW), *Querying*, *Metadata filtering*, *Embedding functions*.
- LangChain Docs — trang tích hợp *Chroma*.
- Paper HNSW gốc: Malkov và Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* (arXiv:1603.09320) — đọc phần hình minh họa là đủ cho tuần này, Tuần 3 đọc kỹ.
