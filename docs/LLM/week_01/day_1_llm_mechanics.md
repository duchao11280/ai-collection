# Ngày 1 — Cơ chế hoạt động của LLM

Bài thực hành đi kèm: gọi API, đếm token và tính chi phí thực của 20 request ([kế hoạch Tuần 1](../learning_plan.md)).
Đọc phần 1–4 trước khi code (khoảng 20 phút); phần 5–6 đọc khi làm tới bước tính chi phí.

!!! abstract "Sau bài này bạn trả lời được"
    - Vì sao cùng một câu hỏi lại tốn số token khác nhau giữa hai model?
    - Context window 8192 token nghĩa là gì, vượt quá thì chuyện gì xảy ra?
    - temperature, top_k, top_p tác động vào bước nào của quá trình sinh chữ?
    - Vì sao token đầu ra đắt hơn token đầu vào?

---

## 1. LLM chỉ làm đúng một việc: đoán token tiếp theo

LLM (Large Language Model — mô hình ngôn ngữ lớn) là một mạng Transformer được huấn luyện để trả lời một câu hỏi duy nhất: *với chuỗi token đã có, phân phối xác suất của token tiếp theo là gì?*

$$P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

Với nền machine learning của bạn, hãy nhìn nó như **một bộ phân loại có số lớp bằng kích thước từ vựng** (khoảng 151 nghìn lớp với Qwen3), được gọi lặp đi lặp lại. Mỗi lần gọi chọn ra đúng một token, token đó được nối vào đầu vào rồi gọi tiếp. Cách sinh này gọi là **tự hồi quy** (autoregressive):

```text
"Thủ đô Việt Nam là"          → model → phân phối xác suất → chọn " Hà"
"Thủ đô Việt Nam là Hà"       → model → phân phối xác suất → chọn " Nội"
"Thủ đô Việt Nam là Hà Nội"   → model → phân phối xác suất → chọn "."
... lặp lại cho tới khi gặp token kết thúc hoặc chạm giới hạn độ dài
```

Quá trình này chia làm hai pha. Gần như mọi khái niệm về hiệu năng và chi phí trong 16 tuần đều bắt nguồn từ đây:

| Pha | Việc làm | Bị giới hạn bởi | Đo trên máy bạn (`qwen3:4b`) |
|---|---|---|---|
| **Prefill** (xử lý prompt) | Đưa toàn bộ prompt qua model **một lượt, song song** | Sức tính toán của GPU | 2558 token trong 1123 mili giây ≈ **2280 token/giây** |
| **Decode** (sinh token) | Sinh **từng token một**, token sau phải chờ token trước | Băng thông bộ nhớ GPU | ≈ **48 token/giây** |

Chênh lệch khoảng **47 lần**. Từ đó suy ra:

- Token đầu ra đắt hơn token đầu vào (mục 6).
- *Time to first token* (thời gian tới token đầu tiên) phụ thuộc chủ yếu vào độ dài prompt; tổng thời gian phụ thuộc chủ yếu vào độ dài câu trả lời.
- Prompt dài nhưng câu trả lời ngắn vẫn nhanh; prompt ngắn nhưng bắt model viết dài thì chậm.

Tuần 9 sẽ đi sâu vào hai pha này cùng KV cache (Key-Value cache — bộ nhớ đệm khóa và giá trị của attention).

---

## 2. Token và tokenizer

### 2.1. Vì sao không cắt theo ký tự hay theo từ

| Cách cắt | Kích thước từ vựng | Độ dài chuỗi | Vấn đề |
|---|---|---|---|
| Theo ký tự | Nhỏ | Rất dài | Attention có chi phí tăng theo bình phương độ dài chuỗi → chậm, đắt |
| Theo từ | Khổng lồ | Ngắn | Gặp từ mới (tên riêng, lỗi chính tả, mã sản phẩm) là không biểu diễn được |
| **Theo mảnh từ (subword)** | Vừa phải (30–250 nghìn) | Vừa phải | Điểm cân bằng: từ phổ biến là một token, từ hiếm ghép từ nhiều mảnh |

### 2.2. BPE — thuật toán tạo từ vựng phổ biến nhất

BPE (Byte Pair Encoding — mã hóa cặp byte) học từ vựng như sau:

1. Bắt đầu với các đơn vị nhỏ nhất. Với *byte-level BPE* đó là 256 giá trị byte.
2. Đếm cặp đơn vị đứng cạnh nhau xuất hiện nhiều nhất trong dữ liệu, gộp cặp đó thành một đơn vị mới.
3. Lặp lại tới khi từ vựng đủ kích thước mong muốn.

Hệ quả cần nhớ:

- Byte-level BPE (dùng trong GPT, Qwen, Llama 3) **không bao giờ gặp token lạ**: chữ nào cũng quy về được byte.
- Tokenizer nén tốt ngôn ngữ nào phụ thuộc vào **tỷ lệ ngôn ngữ đó trong dữ liệu dùng để học từ vựng**. Tiếng Anh và tiếng Trung chiếm phần lớn → được nén tốt. Tiếng Việt ít hơn → tốn nhiều token hơn cho cùng lượng thông tin.
- **Mỗi họ model có tokenizer riêng**: GPT, Claude, Qwen, Llama, Gemma đều khác nhau. `bge-m3` dùng tokenizer SentencePiece của XLM-RoBERTa (khoảng 250 nghìn token, phủ hơn 100 ngôn ngữ).

### 2.3. Tiếng Việt tốn token hơn — số đo thật

Hai câu cùng nghĩa, gửi qua Ollama trên máy bạn:

| Câu | Số ký tự | Token `qwen3:4b` (đã trừ khung chat) | Token `bge-m3` (gồm 2 token đặc biệt) |
|---|---|---|---|
| Hà Nội là thủ đô của Việt Nam, nổi tiếng với phố cổ và ẩm thực đường phố. | 73 | ≈ 22 | 24 |
| Hanoi is the capital of Vietnam, famous for its old quarter and street food. | 76 | ≈ 17 | 19 |

Ở ví dụ này tiếng Việt tốn hơn khoảng 30% token. Tỷ lệ thay đổi theo văn bản và theo tokenizer, nên **đừng dùng quy tắc "1 token ≈ 4 ký tự" của tiếng Anh cho tiếng Việt** — hãy đếm bằng chính tokenizer của model, hoặc đọc số token server trả về.

### 2.4. Cái bẫy Unicode: NFC và NFD

Chữ "ệ" có thể được lưu theo hai cách, hiển thị giống hệt nhau trên màn hình:

- **NFC** (Normalization Form Canonical Composition — dạng dựng sẵn): 1 ký tự `ệ`.
- **NFD** (Normalization Form Canonical Decomposition — dạng tổ hợp): 3 ký tự `e` + dấu mũ + dấu nặng.

Văn bản copy từ macOS, từ một số file PDF hoặc gõ bằng bộ gõ cũ hay ở dạng NFD. Đo lại câu tiếng Việt phía trên:

| Dạng | Số ký tự | Số byte UTF-8 | Token `qwen3:4b` (cả khung chat) | Token `bge-m3` |
|---|---|---|---|---|
| NFC | 73 | 106 | 32 | 24 |
| NFD | 102 | 133 | **92** | 24 |

- `qwen3:4b` tốn **gần gấp 3 lần** token cho cùng một câu nhìn y hệt → gấp 3 chi phí, chiếm gấp 3 context, và model hiểu kém hơn vì hiếm khi thấy dạng này lúc huấn luyện.
- `bge-m3` không bị ảnh hưởng (vector của hai dạng có cosine bằng 1.0) vì tokenizer SentencePiece tự chuẩn hóa đầu vào. Nhưng đó là đặc điểm riêng của model này, không phải quy luật.

**Quy tắc:** chuẩn hóa NFC mọi văn bản trước khi đưa vào pipeline:

```python
import unicodedata
clean_text = unicodedata.normalize("NFC", raw_text)
```

Tuần 2 – Ngày 2 sẽ đưa bước này vào pipeline làm sạch dữ liệu.

### 2.5. Khung chat (chat template) — những token bạn không nhìn thấy

API nhận một danh sách message, nhưng model chỉ đọc **một chuỗi token duy nhất**. Server ghép các message thành chuỗi theo *chat template* của model. Với Qwen (định dạng ChatML):

```text
<|im_start|>system
Bạn là trợ lý kỹ thuật.<|im_end|>
<|im_start|>user
Thủ đô Việt Nam?<|im_end|>
<|im_start|>assistant
```

Gửi một message chỉ có chữ "a" tới `qwen3:4b`, Ollama báo `prompt_eval_count = 11` → khoảng **10 token khung** cho mỗi request. Ngoài ra còn những phần "vô hình" khác cũng bị tính là token đầu vào:

- System prompt mặc định của nhà cung cấp hoặc trong Modelfile.
- Định nghĩa tool: JSON schema của vài tool có thể tốn hàng trăm token (Tuần 6).
- Toàn bộ lịch sử hội thoại, gửi lại ở **mỗi** lượt.

!!! question "Câu hỏi tự kiểm tra của tuần: vì sao cùng một câu hỏi lại tốn số token khác nhau giữa 2 model?"
    1. **Tokenizer khác nhau**: từ vựng và quy tắc gộp khác → cùng văn bản bị cắt thành số mảnh khác nhau, đặc biệt rõ với tiếng Việt.
    2. **Chat template khác nhau**: số token khung cho mỗi message khác.
    3. **Phần ẩn khác nhau**: system prompt mặc định, định nghĩa tool, cách mã hóa ảnh.
    4. **Token suy luận (thinking)**: model có chế độ suy nghĩ sinh thêm hàng trăm tới hàng nghìn token đầu ra, vẫn bị tính tiền và vẫn chiếm context.

---

## 3. Context window

Context window (cửa sổ ngữ cảnh) là **tổng số token tối đa model xử lý trong một lần**, gồm cả đầu vào lẫn đầu ra:

$$N_{\text{system}} + N_{\text{lịch sử}} + N_{\text{tài liệu}} + N_{\text{câu hỏi}} + N_{\text{thinking}} + N_{\text{câu trả lời}} \le \text{context window}$$

Hai con số cần phân biệt:

| Khái niệm | Trên máy bạn | Ai quyết định |
|---|---|---|
| Context tối đa model hỗ trợ | 262144 token | Nhóm huấn luyện model (`ollama show qwen3:4b`) |
| Context thực tế đang chạy (`num_ctx`) | 8192 token | Bạn, qua `OLLAMA_CONTEXT_LENGTH` hoặc tham số `num_ctx` — bị giới hạn bởi VRAM (bộ nhớ GPU) vì KV cache tăng tuyến tính theo độ dài |

**Con số nhỏ hơn mới là giới hạn thật.** Công thức tính VRAM cho KV cache có trong [ollama_setup.md](../ollama_setup.md).

Khi vượt giới hạn:

- API trả phí thường **báo lỗi** → dễ phát hiện.
- Ollama có thể **tự cắt bớt** (bỏ các message cũ, hoặc cắt phần đầu của prompt quá dài) mà không báo lỗi cho client → nguy hiểm hơn nhiều. Trong RAG, tài liệu hoặc chỉ dẫn bị cắt mất, model trả lời như chưa từng đọc. Hãy đếm token trước khi gửi và luôn đặt `num_ctx` rõ ràng.

Context dài hơn **không** có nghĩa là tốt hơn:

- Thời gian prefill và chi phí tăng theo số token.
- Model khai thác thông tin nằm giữa context kém hơn thông tin ở đầu và cuối (hiện tượng *lost in the middle*, Tuần 4).
- Với RAG: đưa 5 đoạn đúng tốt hơn đưa 50 đoạn "cho chắc".

---

## 4. Sampling — từ xác suất tới chữ

Ở mỗi bước, model trả về *logits* $z_i$ (điểm thô chưa chuẩn hóa) cho mọi token trong từ vựng. Bộ lấy mẫu (sampler) biến các điểm này thành **một** token cụ thể. Mọi tham số bạn chỉnh trong mục này đều tác động vào bước lấy mẫu — **không tham số nào làm model thông minh hơn**, chúng chỉ thay đổi cách chọn từ một phân phối có sẵn.

### 4.1. Temperature

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Ví dụ ba token ứng viên có logits $[2.0,\ 1.0,\ 0.5]$:

| Temperature $T$ | Xác suất token 1 | Token 2 | Token 3 | Nhận xét |
|---|---|---|---|---|
| 0.5 | 0.844 | 0.114 | 0.042 | Phân phối nhọn, gần như luôn chọn token 1 |
| 1.0 | 0.629 | 0.231 | 0.140 | Phân phối gốc của model |
| 2.0 | 0.481 | 0.292 | 0.227 | Phân phối phẳng, câu trả lời đa dạng nhưng dễ lạc đề |

- $T \to 0$: luôn chọn token có xác suất cao nhất, gọi là *greedy decoding* (giải mã tham lam).
- **temperature = 0 không đảm bảo kết quả giống hệt nhau** trên server thật: phép cộng số thực trên GPU phụ thuộc thứ tự tính, và cách gom request thành batch thay đổi theo tải. Trên Ollama, cố định thêm `seed` sẽ ổn định hơn.

### 4.2. top_k, top_p, min_p — cắt bỏ phần đuôi của phân phối

Lấy phân phối ở $T = 1$ phía trên: $[0.629,\ 0.231,\ 0.140]$.

| Tham số | Cách làm | Kết quả với ví dụ |
|---|---|---|
| `top_k = 2` | Giữ k token có xác suất cao nhất | Giữ token 1 và 2, chuẩn hóa lại thành $[0.731,\ 0.269]$ |
| `top_p = 0.8` (nucleus sampling) | Giữ tập nhỏ nhất có tổng xác suất ≥ p | 0.629 < 0.8; 0.629 + 0.231 = 0.86 ≥ 0.8 → giữ token 1 và 2 |
| `min_p = 0.3` | Giữ token có xác suất ≥ min_p × xác suất lớn nhất | Ngưỡng 0.3 × 0.629 ≈ 0.189 → giữ token 1 và 2 |

`top_p` và `min_p` hơn `top_k` ở chỗ **tự co giãn**: khi model chắc chắn thì chỉ còn 1–2 ứng viên, khi phân vân thì giữ nhiều hơn. Thứ tự áp dụng các bước (cắt đuôi trước hay chia temperature trước) khác nhau giữa các engine, nên **mỗi lần chỉ chỉnh một tham số** rồi đo kết quả.

### 4.3. Giá trị mặc định của qwen3:4b và cách chỉnh

`ollama show qwen3:4b` cho thấy Modelfile đặt sẵn `temperature 0.6`, `top_k 20`, `top_p 0.95`, `repeat_penalty 1`. Đây là bộ giá trị nhóm Qwen khuyến nghị cho chế độ thinking. Model card của Qwen3 cũng dặn **không dùng greedy decoding khi bật thinking** vì model dễ lặp vô hạn.

Điểm xuất phát theo loại việc (sau đó đo trên dữ liệu thật):

| Loại việc | temperature | Lý do |
|---|---|---|
| Trích xuất thông tin, phân loại, xuất JSON | 0 – 0.3 | Cần ổn định, lặp lại được |
| Hỏi đáp RAG, tóm tắt | 0.2 – 0.7 | Bám tài liệu nhưng câu văn tự nhiên |
| Viết sáng tạo, brainstorm | 0.8 – 1.0 | Cần đa dạng |

Các tham số phạt lặp — `repeat_penalty` (Ollama), `presence_penalty` và `frequency_penalty` (chuẩn OpenAI) — giảm xác suất của các token đã xuất hiện. Dùng khi model lặp câu; đặt quá cao sẽ khiến model né cả những từ cần lặp lại như tên riêng, thuật ngữ.

---

## 5. Điều kiện dừng: stop sequence, giới hạn độ dài, lý do kết thúc

Model ngừng sinh khi một trong ba điều xảy ra:

1. **Sinh ra token kết thúc của chính nó.** `ollama show qwen3:4b` liệt kê `stop "<|im_end|>"` và `stop "<|im_start|>"` — chính là các token khung chat ở mục 2.5.
2. **Gặp stop sequence bạn khai báo.** Ví dụ `stop=["\n\n"]` để chỉ lấy một đoạn, hoặc `stop=["Câu hỏi:"]` để model không tự bịa thêm câu hỏi tiếp theo khi dùng few-shot. Chuỗi dừng thường **không** nằm trong kết quả trả về. Cẩn thận: nếu chuỗi dừng có thể xuất hiện trong nội dung hợp lệ (ví dụ `"\n"` khi model viết code), câu trả lời sẽ bị cắt sớm.
3. **Chạm giới hạn số token đầu ra:** `num_predict` (Ollama), `max_tokens` (Anthropic, OpenAI).

Luôn kiểm tra **lý do kết thúc**:

| Nguồn | Trường | Giá trị báo bị cắt cụt |
|---|---|---|
| Ollama | `done_reason` | `length` |
| Anthropic | `stop_reason` | `max_tokens` |
| Chuẩn OpenAI | `finish_reason` | `length` |

Một câu trả lời bị cắt cụt nhìn vẫn "có vẻ ổn", nhưng một JSON bị cắt cụt là JSON hỏng — đây là lỗi hay gặp nhất khi làm bài Ngày 2. Với model có thinking, token suy luận cũng tính vào giới hạn đầu ra: đặt `num_predict` quá thấp, model có thể hết lượt ngay trong lúc đang "nghĩ", chưa kịp trả lời.

---

## 6. Chi phí

### 6.1. Công thức

$$\text{Chi phí} = \frac{N_{\text{vào}} \times P_{\text{vào}} + N_{\text{ra}} \times P_{\text{ra}}}{10^6}$$

với $P$ là giá cho mỗi 1 triệu token. Giá đầu ra thường gấp 3–5 lần giá đầu vào — bảng prefill/decode ở mục 1 giải thích vì sao: một token đầu ra chiếm GPU lâu hơn nhiều so với một token đầu vào.

Ví dụ với **giá giả định** $P_{\text{vào}} = 3$ USD và $P_{\text{ra}} = 15$ USD (giá thật xem trên trang của nhà cung cấp, thay đổi thường xuyên). 20 request, mỗi request 1200 token vào và 400 token ra:

| Phần | Số token | Chi phí |
|---|---|---|
| Đầu vào | 20 × 1200 = 24 000 | 24 000 × 3 / 10⁶ = 0.072 USD |
| Đầu ra | 20 × 400 = 8 000 | 8 000 × 15 / 10⁶ = 0.120 USD |
| **Tổng** | 32 000 | **0.192 USD** |

Đầu ra chỉ chiếm 25% số token nhưng chiếm **62.5% chi phí**.

### 6.2. Những khoản hay bị quên

- **Lịch sử hội thoại.** Mỗi lượt gửi lại toàn bộ lịch sử. Hội thoại 10 lượt, mỗi lượt (hỏi cộng đáp) 300 token: lượt thứ $i$ gửi khoảng $300 \times i$ token → tổng $300 \times (1 + 2 + \dots + 10) = 16\,500$ token đầu vào, trong khi nội dung thật chỉ có 3000 token. Chi phí tăng theo **bình phương** số lượt.
- **Token thinking** được tính như token đầu ra.
- **Định nghĩa tool và system prompt** nằm trong đầu vào của *mọi* request.
- **Gửi lại vì kết quả không đạt** (ví dụ JSON sai schema): mỗi lần thử lại là một request tính tiền đầy đủ. Tỷ lệ thành công 90% nghĩa là chi phí thực tế cao hơn khoảng 11% (vì $1 / 0.9 \approx 1.11$).
- **Prompt caching** (Ngày 2): phần đầu prompt lặp lại được tính giá rẻ hơn nhiều.
- **Batch API**: nhiều nhà cung cấp giảm giá nếu bạn chấp nhận nhận kết quả chậm (thường trong vòng 24 giờ).

### 6.3. Chạy local không có nghĩa là miễn phí

Với Ollama, đơn vị "tiền" là **thời gian và VRAM**. Tốc độ decode khoảng 48 token/giây nghĩa là một câu trả lời 400 token mất khoảng 8 giây; 1000 request như vậy chiếm GPU hơn 2 giờ. Tuần 10 sẽ tính điểm hòa vốn giữa tự host và gọi API.

### 6.4. Đọc số liệu ở đâu

| Nguồn | Token vào | Token ra | Thời gian |
|---|---|---|---|
| Ollama `/api/chat` | `prompt_eval_count` | `eval_count` (gồm cả thinking) | `prompt_eval_duration`, `eval_duration`, `load_duration`, `total_duration` — đơn vị nano giây |
| Anthropic | `usage.input_tokens` (cộng `cache_read_input_tokens`, `cache_creation_input_tokens`) | `usage.output_tokens` | Tự đo |
| Chuẩn OpenAI (gồm cả endpoint `/v1` của Ollama) | `usage.prompt_tokens` | `usage.completion_tokens` | Tự đo |

Khung tính toán cho bài thực hành (kết quả của `POST /api/chat` với `"stream": False`):

```python
PRICE_INPUT_PER_MILLION = 3.0     # giá giả định, thay bằng giá thật
PRICE_OUTPUT_PER_MILLION = 15.0

data = response.json()
prompt_tokens = data["prompt_eval_count"]
output_tokens = data["eval_count"]
prefill_speed = prompt_tokens / (data["prompt_eval_duration"] / 1e9)   # token/giây
decode_speed = output_tokens / (data["eval_duration"] / 1e9)          # token/giây
cost = (prompt_tokens * PRICE_INPUT_PER_MILLION + output_tokens * PRICE_OUTPUT_PER_MILLION) / 1_000_000
finished_normally = data.get("done_reason") == "stop"
```

---

## 7. Câu hỏi tự kiểm tra

1. Prompt 3000 token, câu trả lời 300 token, chạy trên máy bạn. Ước lượng tổng thời gian (bỏ qua thời gian nạp model).
2. Đặt temperature = 0, gọi API trả phí hai lần với cùng prompt, nhận hai câu trả lời hơi khác nhau. Có phải lỗi không?
3. Vì sao `top_p = 0.1` cho kết quả gần giống greedy decoding?
4. Model trả về JSON bị cụt ở giữa. Trường nào cần kiểm tra đầu tiên?
5. Chatbot 20 lượt, mỗi lượt (hỏi cộng đáp) 500 token. Tổng token đầu vào đã gửi là bao nhiêu?

### Gợi ý đáp án

1. Prefill ≈ 3000 / 2280 ≈ 1.3 giây; decode ≈ 300 / 48 ≈ 6.3 giây → **khoảng 7.6 giây**. Pha decode chiếm hơn 80% — muốn nhanh hơn thì rút ngắn câu trả lời trước khi rút ngắn prompt.
2. Không. Tính toán số thực trên GPU phụ thuộc thứ tự cộng và cách server gom batch, nên temperature = 0 chỉ "gần như" tất định.
3. Token có xác suất cao nhất gần như luôn vượt 0.1, nên tập nhỏ nhất có tổng ≥ 0.1 chỉ còn đúng token đó.
4. `done_reason` (Ollama) hoặc `finish_reason` / `stop_reason`. Nếu là `length` / `max_tokens` → tăng `num_predict` hoặc yêu cầu câu trả lời ngắn hơn. Nhớ rằng token thinking cũng ăn vào giới hạn này.
5. $500 \times (1 + 2 + \dots + 20) = 500 \times 210 = 105\,000$ token, trong khi nội dung thật chỉ có $500 \times 20 = 10\,000$ token.

---

## Đọc thêm

- Anthropic Docs — *Token counting* và *Pricing*: cách nhà cung cấp tính token, kể cả tool và ảnh.
- Hugging Face NLP Course — chương *Tokenizers*: BPE, WordPiece, Unigram.
- Model card của Qwen3 trên Hugging Face — mục *Best Practices* về tham số sampling.
- Tuần 9 – Ngày 1 trong [kế hoạch](../learning_plan.md): prefill, decode, KV cache ở mức chi tiết.
