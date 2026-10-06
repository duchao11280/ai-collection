# Ngày 2 — Prompt engineering và structured output

Bài thực hành đi kèm: ép model trả JSON đúng schema 20/20 lần ([kế hoạch Tuần 1](../learning_plan.md)).
Đọc phần 1–5 trước khi code; phần 6 (prompt caching) đọc khi làm phần mở rộng.

!!! abstract "Sau bài này bạn trả lời được"
    - Viết prompt là "ra lệnh" hay là việc gì, xét theo cơ chế ở Ngày 1?
    - Few-shot tác động vào model thế nào, và những thiên lệch nào đi kèm?
    - JSON mode khác constrained decoding theo schema ở đâu? Vì sao đã có constrained decoding vẫn cần Pydantic?
    - Vì sao prompt caching yêu cầu phần đầu prompt giống hệt nhau, và nên sắp xếp prompt thế nào?

---

## 1. Prompt dưới góc nhìn cơ chế

Từ Ngày 1: model chỉ nhìn thấy một chuỗi token và đoán tiếp. Viết prompt là **đặt điều kiện cho phân phối xác suất** đó, sao cho câu trả lời bạn muốn trở thành phần tiếp nối có xác suất cao nhất. Không có "lệnh" nào được thực thi theo nghĩa lập trình — mọi thứ đều chỉ là ngữ cảnh.

Các vai trò (role) trong một request chat:

| Vai trò | Nội dung | Ghi chú |
|---|---|---|
| `system` | Vai trò, quy tắc, định dạng đầu ra, kiến thức nền cố định | Model được huấn luyện để ưu tiên chỉ dẫn ở đây; đặt ở đầu nên được hưởng prompt caching |
| `user` | Yêu cầu cụ thể, dữ liệu đầu vào | Phần thay đổi theo từng request |
| `assistant` | Câu trả lời trước đó của model, hoặc câu trả lời mẫu bạn tự viết | Dùng cho lịch sử hội thoại và few-shot |
| `tool` | Kết quả trả về của tool | Tuần 6 |

---

## 2. Nguyên tắc viết prompt

1. **Rõ ràng và cụ thể như viết cho một đồng nghiệp mới vào**: rất giỏi, nhưng không biết gì về dự án và không đọc được suy nghĩ của bạn.
2. **Giải thích lý do, không chỉ ra lệnh.** "Không dùng markdown vì kết quả sẽ được đọc bằng giọng nói" giúp model tự suy ra cả những trường hợp bạn chưa liệt kê.
3. **Nói điều cần làm thay vì điều cấm.** "Trả lời bằng một đoạn văn liền mạch" hiệu quả hơn "Đừng dùng gạch đầu dòng".
4. **Tách chỉ dẫn và dữ liệu bằng thẻ.** Đặt tài liệu trong `<tai_lieu>...</tai_lieu>`, câu hỏi trong `<cau_hoi>...</cau_hoi>`. Model phân biệt được đâu là yêu cầu, đâu là nội dung cần xử lý. Đây cũng là lớp phòng thủ đầu tiên trước prompt injection (Tuần 8 và Tuần 14).
5. **Mô tả định dạng đầu ra cụ thể**, và nói rõ phải làm gì khi không có câu trả lời.
6. **Prompt là code**: lưu trong file, quản lý bằng git, mỗi lần sửa thì chạy lại trên cùng một bộ đầu vào kiểm tra.

So sánh:

```text
# Prompt kém
Tóm tắt văn bản này.

# Prompt tốt
Bạn tóm tắt biên bản họp cho người quản lý không dự họp.
Họ cần biết: quyết định đã chốt, việc cần làm (ai làm, hạn khi nào), vấn đề còn bỏ ngỏ.

<bien_ban>
{meeting_notes}
</bien_ban>

Viết tối đa 5 gạch đầu dòng, mỗi dòng dưới 25 từ, bằng tiếng Việt.
Nếu biên bản không có quyết định nào, ghi rõ "Chưa có quyết định".
```

---

## 3. Zero-shot và few-shot

- **Zero-shot**: chỉ mô tả nhiệm vụ.
- **Few-shot**: đưa kèm vài cặp ví dụ đầu vào → đầu ra. Model học **định dạng, độ dài, văn phong và tập nhãn** ngay từ ví dụ trong ngữ cảnh (*in-context learning*), không cập nhật trọng số nào.

Với chat model, cách đưa ví dụ tự nhiên nhất là các lượt hội thoại mẫu:

```python
messages = [
    {"role": "system", "content": "Phân loại phản hồi khách hàng thành đúng một nhãn: khen, chê, hỏi, khác."},
    {"role": "user", "content": "Giao hàng nhanh, đóng gói cẩn thận."},
    {"role": "assistant", "content": "khen"},
    {"role": "user", "content": "Cửa hàng có giao ra Côn Đảo không?"},
    {"role": "assistant", "content": "hỏi"},
    {"role": "user", "content": "Áo mới mặc một lần đã bai cổ."},
    {"role": "assistant", "content": "chê"},
    {"role": "user", "content": text_to_classify},
]
```

Những điều cần biết về few-shot:

- **Model bắt chước cả những gì bạn không cố ý.** Ví dụ đều ngắn → câu trả lời ngắn. Phần lớn ví dụ là "khen" → model nghiêng về "khen" (thiên lệch nhãn đa số). Ví dụ đặt cuối cùng có ảnh hưởng mạnh nhất (thiên lệch gần nhất).
- **Đa dạng và phủ trường hợp biên**: có ví dụ khó, ví dụ thuộc nhãn "khác", ví dụ không trả lời được.
- **3–5 ví dụ** thường là đủ; nhiều hơn tốn token mà ít cải thiện.
- **Model nhỏ như `qwen3:4b` hưởng lợi từ few-shot nhiều hơn model lớn** — nó cần được "chỉ tận tay" định dạng.
- Ví dụ lặp lại ở mọi request → đặt ở phần đầu prompt để được cache (mục 6).

---

## 4. Suy luận trước khi trả lời

- Yêu cầu model suy nghĩ từng bước trước khi kết luận (*chain-of-thought*) giúp các bài toán nhiều bước. Giải thích theo cơ chế Ngày 1: mỗi token sinh ra là thêm một lượt tính toán, và phần kết luận được sinh **sau** phần suy luận nên được điều kiện hóa bởi phần đó.
- `qwen3:4b` có chế độ **thinking** tích hợp: bật bằng `think: true`, phần suy luận nằm ở trường riêng `message.thinking` (xem `llm-lab/scripts/hello_llm.py`).
- Đánh đổi: thinking tốn token đầu ra → chậm và đắt. Với trích xuất, phân loại, xuất JSON đơn giản → **tắt** (`think: false`). Với câu hỏi cần lập luận → bật, và **đo** xem có thực sự tốt hơn không.

---

## 5. Structured output — ép model trả đúng định dạng

Đây là nền móng cho tool use (Tuần 6) và cho mọi pipeline tự động: code phía sau phải parse được kết quả.

### 5.1. Bốn mức đảm bảo

| Mức | Cách làm | Đảm bảo được gì | Lỗi vẫn gặp |
|---|---|---|---|
| 1. Chỉ dặn trong prompt | "Trả về JSON với các trường..." | Không đảm bảo gì | Bọc trong khối code markdown, thêm câu dẫn, thiếu trường, sai kiểu, dùng nháy đơn |
| 2. JSON mode | Ollama `format: "json"` | Cú pháp JSON hợp lệ | Sai tên trường, thiếu trường, sai kiểu dữ liệu |
| 3. Constrained decoding theo JSON schema | Ollama `format: <JSON schema>`; `response_format` kiểu `json_schema` ở chuẩn OpenAI | Đúng cấu trúc schema (trong phạm vi các ràng buộc engine hỗ trợ) | Giá trị vẫn có thể sai; JSON bị cắt cụt nếu hết `num_predict` |
| 4. Tool use / function calling | Model sinh tham số gọi tool theo schema | Tương tự mức 3, tùy nhà cung cấp | Model có thể chọn không gọi tool |

Dù ở mức nào cũng phải có **lớp kiểm tra phía sau**: parse → validate bằng Pydantic → kiểm tra logic nghiệp vụ → sai thì thử lại kèm thông báo lỗi.

### 5.2. Constrained decoding hoạt động thế nào

Nhớ lại Ngày 1: ở mỗi bước, sampler chọn một token từ phân phối xác suất. Constrained decoding (giải mã có ràng buộc) chèn thêm một bước vào đó:

1. JSON schema được dịch thành một ngữ pháp (grammar). Ví dụ llama.cpp — nền tảng ban đầu của Ollama — dùng định dạng GBNF (GGML Backus-Naur Form).
2. Ở mỗi bước, engine tính tập token **hợp lệ** theo trạng thái hiện tại của ngữ pháp. Ví dụ ngay sau `{"label": ` thì chỉ được mở chuỗi bằng dấu `"`.
3. Logits của mọi token không hợp lệ bị gán $-\infty$ → xác suất bằng 0 → không thể được chọn.

```text
Đã sinh:   {"label": "
Model muốn: "Khách"  (p = 0.40)   "khen" (p = 0.35)   "chê" (p = 0.15)   "hỏi", "khác" (gần 0) ...
Schema:    label ∈ {khen, chê, hỏi, khác}
Sau mặt nạ: "Khách" → 0   "khen" (0.35 → 0.70)   "chê" (0.15 → 0.30)   rồi chuẩn hóa lại và lấy mẫu như bình thường
```

Hệ quả thực tế:

- **Cú pháp được đảm bảo, nội dung thì không.** Model vẫn có thể điền giá trị bịa — chỉ là bịa đúng định dạng.
- **Thứ tự trường quan trọng**, vì sinh tự hồi quy: trường sinh trước ảnh hưởng trường sinh sau. Muốn model lập luận rồi mới kết luận thì đặt trường `reasoning` **trước** trường `label`.
- **`enum` rất mạnh cho phân loại**: đầu ra bị giới hạn đúng trong tập nhãn cho phép.
- **Vẫn nên mô tả schema trong prompt.** Mặt nạ chỉ chặn token sai, không giúp model hiểu ý nghĩa từng trường. Thêm `description` cho từng trường và nhắc lại yêu cầu bằng lời trong prompt — tài liệu của Ollama cũng khuyên như vậy.
- **Bị cắt cụt vẫn hỏng.** Hết `num_predict` giữa chừng → JSON không được đóng → kiểm tra `done_reason`.
- **Không phải ràng buộc nào của schema cũng được engine thực thi** (ví dụ `maxItems`, `pattern`, `minimum`). Validate phía sau mới là chốt chặn cuối cùng.

### 5.3. Pydantic là "hợp đồng" giữa code và model

```python
from typing import Literal

from pydantic import BaseModel, Field


class FeedbackAnalysis(BaseModel):
    reasoning: str = Field(description="Lý do chọn nhãn, một câu")
    label: Literal["khen", "chê", "hỏi", "khác"]
    keywords: list[str] = Field(description="Tối đa 3 từ khóa", max_length=3)


schema = FeedbackAnalysis.model_json_schema()               # truyền vào "format" của Ollama
result = FeedbackAnalysis.model_validate_json(response_text)  # ném ValidationError nếu sai
```

- `Literal[...]` được dịch thành `enum` trong JSON schema.
- `reasoning` đứng trước `label` theo đúng nguyên tắc thứ tự trường ở mục 5.2.
- Khi validate thất bại, gửi lại cho model **nội dung lỗi của Pydantic** — model sửa theo lỗi cụ thể tốt hơn nhiều so với "hãy thử lại".

### 5.4. Đo độ tin cậy: "20/20" nghĩa là gì

Bài thực hành yêu cầu JSON đúng schema 20/20 lần. Hãy định nghĩa rõ "đúng" theo ba mức:

1. Parse được JSON.
2. Validate được bằng Pydantic.
3. Giá trị hợp lý về nghiệp vụ (ví dụ nhãn khớp với nhãn bạn tự gán tay).

Mức 1–2 đo **độ tin cậy định dạng**; mức 3 đo **chất lượng**. Đây là hai thứ khác nhau: constrained decoding gần như chắc chắn cho 20/20 ở mức 2, nhưng mức 3 thì không. Nên:

- Chạy thử cả ba cách: chỉ dặn prompt → `format: "json"` → `format: schema`, ghi tỷ lệ thành công của từng cách. Đây chính là số liệu cho note hằng ngày.
- Chạy với temperature > 0 (ví dụ 0.7) để lộ ra phương sai; temperature = 0 có thể che mất lỗi.
- Ghi lại lỗi theo loại (thiếu trường, sai kiểu, bị cắt cụt, sai nhãn) để biết cần sửa prompt, sửa schema hay tăng `num_predict`.

---

## 6. Prompt caching

### 6.1. Cơ chế

Trong pha prefill (Ngày 1), model tính KV cache cho từng token của prompt. Nếu request mới có **phần đầu giống hệt từng token** với request trước, server tái sử dụng KV cache của phần đó thay vì tính lại. Hệ quả:

- Cache hoạt động theo **tiền tố** (prefix). Chỉ cần một token khác ở vị trí thứ 10 là mọi thứ từ vị trí đó trở đi phải tính lại.
- Sắp xếp prompt: **phần ổn định trước, phần thay đổi sau**.

```text
[định nghĩa tool] → [system prompt] → [ví dụ few-shot] → [tài liệu dùng chung] → [lịch sử] → [câu hỏi mới]
 ◄──────────────────── ổn định, được cache ────────────────────►  ◄───────── thay đổi ─────────►
```

- Kẻ phá cache kinh điển: chèn thời gian hiện tại (`Bây giờ là 2026-09-26 23:28:30`) hoặc mã request vào **đầu** system prompt. Nếu thật sự cần, đặt thông tin đó ở cuối.

### 6.2. Đo trên máy bạn

System prompt dài 2558 token, gửi liên tiếp ba câu hỏi khác nhau tới `qwen3:4b`:

| Lần gọi | Câu hỏi | `prompt_eval_count` | Thời gian prefill |
|---|---|---|---|
| 1 | Thủ đô Việt Nam? | 2558 | 1123 mili giây |
| 2 | Thủ đô Nhật Bản? | 2558 | 39.5 mili giây |
| 3 | Thủ đô Pháp? | 2557 | 30.8 mili giây |

- Nhanh hơn khoảng **30 lần** dù câu hỏi thay đổi — vì phần đầu giống nhau.
- Để ý: Ollama vẫn báo đủ 2558 token. **Đừng nhìn số token để biết có trúng cache hay không, hãy nhìn thời gian.** Công thức tốc độ prefill ở Ngày 1 sẽ ra con số ảo khi trúng cache.
- Cache local mất khi model bị gỡ khỏi bộ nhớ (mặc định sau 5 phút không dùng) hoặc khi một prompt khác chiếm chỗ.

### 6.3. Prompt caching ở nhà cung cấp API

Mỗi nhà cung cấp làm một kiểu: có nơi tự động cache khi prompt đủ dài, có nơi bạn phải tự đánh dấu. Ví dụ với Anthropic (theo tài liệu chính thức, kiểm tra ngày 2026-09-26):

- Đánh dấu điểm cache bằng `cache_control`, tối đa 4 điểm mỗi request. Thứ tự tiền tố: `tools` → `system` → `messages`.
- Thời gian sống mặc định 5 phút, được làm mới miễn phí mỗi lần trúng cache; có tùy chọn 1 giờ.
- Giá so với token đầu vào thông thường: **ghi cache** gấp 1.25 lần (loại 5 phút) hoặc 2 lần (loại 1 giờ); **đọc cache** chỉ bằng 0.1 lần (một số model còn thấp hơn).
- Có độ dài tối thiểu, từ 512 tới 4096 token tùy model. Ngắn hơn thì **không được cache và cũng không báo lỗi**.
- Đọc kết quả ở `usage.cache_creation_input_tokens` và `usage.cache_read_input_tokens`. Lúc này `usage.input_tokens` chỉ còn phần nằm sau điểm cache cuối cùng.

Điểm hòa vốn với cache 5 phút, gọi $n$ lần cùng một tiền tố trong thời gian sống của cache (tính theo đơn vị "giá của tiền tố khi không cache"):

$$\text{Không cache: } n \times 1 \qquad\qquad \text{Có cache: } 1.25 + (n - 1) \times 0.1$$

- $n = 1$: 1 so với 1.25 → cache **làm đắt thêm** 25%.
- $n = 2$: 2 so với 1.35 → **đã có lời** từ lần gọi thứ hai.
- $n = 10$: 10 so với 2.15 → tiết kiệm khoảng 78% chi phí phần tiền tố, chưa kể latency giảm mạnh.

---

## 7. Câu hỏi tự kiểm tra

1. Vì sao nên đặt trường `reasoning` trước trường `label` trong schema?
2. JSON mode khác constrained decoding theo schema ở điểm nào? Cho một ví dụ JSON mode trả về JSON hợp lệ nhưng vẫn "sai".
3. System prompt bắt đầu bằng dòng `Thời gian hiện tại: {now}`. Có vấn đề gì?
4. Đã dùng constrained decoding, tại sao vẫn cần Pydantic?
5. Một tiền tố 10 000 token chỉ được dùng đúng một lần. Có nên bật prompt caching của Anthropic cho nó không?

### Gợi ý đáp án

1. Sinh tự hồi quy: `label` được sinh sau nên được điều kiện hóa bởi phần lập luận. Đặt ngược lại thì model chốt nhãn trước rồi mới viết lý do để hợp thức hóa.
2. JSON mode chỉ đảm bảo cú pháp. Ví dụ `{"nhan": "tích cực"}` là JSON hợp lệ nhưng sai tên trường và giá trị không thuộc tập nhãn. Constrained decoding theo schema chặn được cả hai lỗi này.
3. Thời gian đổi ở mỗi request → tiền tố khác ngay từ đầu → không bao giờ trúng cache, mọi request phải prefill lại toàn bộ.
4. Vì (a) nội dung vẫn có thể sai về nghiệp vụ, (b) engine có thể không thực thi hết ràng buộc của schema, (c) JSON có thể bị cắt cụt khi hết giới hạn token, (d) đổi model hoặc nhà cung cấp thì mức đảm bảo cũng đổi theo.
5. Không. Ghi cache tốn 1.25 lần giá thường mà không có lần đọc nào bù lại → đắt thêm 25%.

---

## Đọc thêm

- Anthropic Docs — *Prompt engineering overview* và *Prompt caching*.
- Ollama Docs — *Structured outputs*.
- Pydantic Docs — *JSON Schema* và *Validation errors*.
- [ollama_setup.md](../ollama_setup.md) mục 4.3 — ví dụ structured output với Ollama và Pydantic.
