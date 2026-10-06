# Ngày 3 — LangChain core: Runnable và LCEL

Bài thực hành đi kèm: build chain tóm tắt có streaming ([kế hoạch Tuần 1](../learning_plan.md)).
Cài đặt: `pip install langchain-core langchain-ollama` trong môi trường conda `llm`.

!!! abstract "Sau bài này bạn trả lời được"
    - `Runnable` là gì và vì sao mọi thứ trong LangChain đều là `Runnable`?
    - Trong `prompt | model | parser`, dữ liệu ở mỗi mũi tên có kiểu gì?
    - `invoke`, `stream`, `batch` khác nhau về bản chất ở đâu, và khi nào `batch` **không** nhanh hơn?
    - Viết một chain 3 bước mà không cần copy-paste như thế nào?

---

## 1. LangChain giải quyết vấn đề gì

Khi tự viết bằng `httpx` như `llm-lab/scripts/hello_llm.py`, bạn tự lo mọi thứ: định dạng message, parse kết quả, streaming, chạy song song, thử lại khi lỗi, đổi nhà cung cấp. LangChain chuẩn hóa các việc đó sau **một giao diện chung** tên là `Runnable`. Đổi `ChatOllama` sang `ChatAnthropic` thì phần còn lại của chain giữ nguyên.

Các gói (package) bạn sẽ gặp:

| Gói | Chứa gì | Dùng từ |
|---|---|---|
| `langchain-core` | Giao diện `Runnable`, prompt template, message, output parser | Tuần 1 |
| `langchain-ollama`, `langchain-anthropic`, `langchain-openai` | Kết nối tới từng nhà cung cấp: `ChatOllama`, `OllamaEmbeddings`... | Tuần 1 |
| `langchain-chroma`, `langchain-qdrant` | Kết nối vector database | Tuần 1, Tuần 3 |
| `langchain` | Thành phần cấp cao, đặc biệt là agent | Tuần 6 |
| `langgraph` | Điều phối agent dạng đồ thị có trạng thái | Tuần 6–8 |

!!! note "Về phiên bản"
    Từ LangChain 1.0 (cuối năm 2025), trọng tâm của gói `langchain` chuyển sang agent chạy trên LangGraph. Nhưng `Runnable` và LCEL trong `langchain-core` vẫn là nền móng: chat model, retriever, node của LangGraph đều là `Runnable`. Kiến thức hôm nay không bị lỗi thời.

Khi nào **không** cần LangChain: script một bước, cần kiểm soát tuyệt đối request và response, hoặc khi lớp trừu tượng che mất lỗi. Tuần 6 bạn sẽ tự viết vòng lặp agent không dùng framework — cách tốt nhất để hiểu framework làm hộ mình những gì.

---

## 2. Runnable — giao diện chung

Mọi thành phần (prompt, model, parser, retriever, hàm tự viết) đều là `Runnable` và có chung bộ phương thức:

| Phương thức | Đầu vào | Đầu ra | Khi nào dùng |
|---|---|---|---|
| `invoke(x)` | Một đầu vào | Một đầu ra, chờ xong mới trả | Script, xử lý nền |
| `stream(x)` | Một đầu vào | Iterator trả từng mảnh (chunk) | Giao diện chat, cần hiện chữ ngay |
| `batch([x1, x2, ...])` | Danh sách | Danh sách đầu ra, đúng thứ tự đầu vào | Xử lý hàng loạt |
| `ainvoke`, `astream`, `abatch` | Như trên | Phiên bản bất đồng bộ (async) | Web server (Tuần 11) |
| `astream_events` | Một đầu vào | Sự kiện của mọi bước con | Hiển thị tiến trình, debug |

Kiểu dữ liệu vào và ra của các thành phần hay dùng — nắm bảng này thì nối chain không bị lỗi kiểu:

| Thành phần | Nhận | Trả |
|---|---|---|
| `ChatPromptTemplate` | `dict` chứa các biến | `PromptValue` (bọc danh sách message) |
| `ChatOllama` (chat model) | `PromptValue`, danh sách message, hoặc `str` | `AIMessage` (khi stream: `AIMessageChunk`) |
| `StrOutputParser` | `AIMessage` | `str` |
| `JsonOutputParser` | `AIMessage` | `dict` |
| Retriever | `str` (câu hỏi) | `list[Document]` |
| Hàm Python bọc trong `RunnableLambda` | Tùy bạn | Tùy bạn |

---

## 3. LCEL và toán tử pipe

LCEL (LangChain Expression Language — ngôn ngữ biểu thức của LangChain) ghép các `Runnable` bằng toán tử `|`: đầu ra của bước trước là đầu vào của bước sau, giống pipe trong shell.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen3:4b", temperature=0.2, num_ctx=8192, reasoning=False)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn tóm tắt văn bản kỹ thuật bằng tiếng Việt, tối đa 3 câu."),
    ("human", "{text}"),
])

chain = prompt | model | StrOutputParser()
#       dict ──► PromptValue ──► AIMessage ──► str
summary = chain.invoke({"text": "..."})
```

Bên dưới, `a | b` gọi phương thức `a.__or__(b)` và tạo ra một `RunnableSequence`. **Bản thân chain cũng là một `Runnable`**, nên nó có sẵn `invoke`, `stream`, `batch` và có thể lồng vào một chain khác như một bước bình thường.

Khi ghép bằng `|`, LangChain tự chuyển kiểu (coercion):

- Một `dict` → `RunnableParallel`: chạy các nhánh **song song** trên cùng một đầu vào, gom kết quả thành `dict`.
- Một hàm Python → `RunnableLambda`.

Các khối xây dựng hay dùng:

| Khối | Tác dụng |
|---|---|
| `RunnablePassthrough()` | Chuyển nguyên đầu vào sang bước sau |
| `RunnablePassthrough.assign(key=runnable)` | Giữ nguyên `dict` đầu vào, thêm một khóa mới được tính từ chính `dict` đó |
| `RunnableParallel(first=..., second=...)` | Chạy nhiều nhánh song song, trả `dict` |
| `RunnableLambda(function)` | Bọc hàm thường thành `Runnable` |
| `itemgetter("key")` (thư viện chuẩn `operator`) | Lấy một khóa trong `dict`, hay dùng ở đầu một nhánh |

### 3.1. Chain 3 bước tuần tự

Câu hỏi tự kiểm tra của tuần yêu cầu tự viết được chain 3 bước. Mẫu dưới đây: tóm tắt → trích từ khóa từ bản tóm tắt → viết tiêu đề từ cả tóm tắt lẫn từ khóa. Điểm mấu chốt là **giữ lại kết quả của các bước trước** bằng `RunnablePassthrough.assign`:

```python
from langchain_core.runnables import RunnablePassthrough

summarize = (
    ChatPromptTemplate.from_template("Tóm tắt đoạn sau trong 2 câu:\n{text}")
    | model | StrOutputParser()
)
extract_keywords = (
    ChatPromptTemplate.from_template("Liệt kê 3 từ khóa, cách nhau bằng dấu phẩy:\n{summary}")
    | model | StrOutputParser()
)
make_title = (
    ChatPromptTemplate.from_template("Viết một tiêu đề dưới 12 từ.\nTóm tắt: {summary}\nTừ khóa: {keywords}")
    | model | StrOutputParser()
)

chain = (
    RunnablePassthrough.assign(summary=summarize)            # {text} → {text, summary}
    | RunnablePassthrough.assign(keywords=extract_keywords)  # → {text, summary, keywords}
    | RunnablePassthrough.assign(title=make_title)           # → {text, summary, keywords, title}
)
result = chain.invoke({"text": "..."})
```

Mỗi prompt template chỉ lấy những biến nó cần từ `dict`, các khóa thừa bị bỏ qua.

### 3.2. Hai nhánh song song

```python
from langchain_core.runnables import RunnableParallel

translate = (
    ChatPromptTemplate.from_template("Dịch sang tiếng Anh:\n{text}")
    | model | StrOutputParser()
)
parallel_chain = RunnableParallel(summary=summarize, english=translate)
parallel_chain.invoke({"text": "..."})   # {"summary": "...", "english": "..."}
```

Về lý thuyết, tổng thời gian bằng thời gian của nhánh chậm nhất. Thực tế trên Ollama local còn tùy server có xử lý song song được hay không (mục 6.2).

Mẹo xem cấu trúc chain: `chain.get_graph().print_ascii()` (cần cài thêm gói `grandalf`).

---

## 4. Prompt template

- `PromptTemplate`: một chuỗi văn bản duy nhất. Dùng khi cần chuỗi thuần.
- `ChatPromptTemplate`: danh sách message có vai trò. Dùng với chat model — gần như lúc nào cũng dùng cái này.
- `MessagesPlaceholder("history")`: chèn cả một danh sách message (ví dụ lịch sử hội thoại) vào một vị trí.
- `.partial(language="tiếng Việt")`: điền sẵn một số biến, phần còn lại điền lúc gọi.

**Bẫy dấu ngoặc nhọn**: template dùng `{ten_bien}` để đánh dấu biến. Nếu prompt chứa ví dụ JSON như `{"label": "khen"}`, LangChain sẽ hiểu nhầm đó là biến. Phải nhân đôi dấu ngoặc: `{{"label": "khen"}}`.

**Thói quen debug**: gọi riêng prompt để xem **chính xác** những message sẽ gửi cho model trước khi nối tiếp:

```python
print(prompt.invoke({"text": "Xin chào"}).to_messages())
print(prompt.input_variables)   # danh sách biến template đang chờ
```

---

## 5. Output parser và structured output

| Cách | Cơ chế (liên hệ Ngày 2) | Có stream được không |
|---|---|---|
| `StrOutputParser` | Lấy `.content` của `AIMessage` | Có, từng mảnh chữ |
| `JsonOutputParser` | Parse JSON từ văn bản model sinh ra | Có, trả `dict` "dở dang" lớn dần theo thời gian |
| `PydanticOutputParser` | Chèn hướng dẫn định dạng vào prompt (mức 1 ở Ngày 2) rồi validate | Không có ý nghĩa |
| `model.with_structured_output(Schema)` | Dùng cơ chế gốc của nhà cung cấp: JSON schema hoặc tool calling (mức 3–4 ở Ngày 2) | Tùy nhà cung cấp |

→ Ưu tiên `with_structured_output` khi model hỗ trợ: đảm bảo mạnh hơn vì dùng constrained decoding thay vì chỉ dặn trong prompt.

```python
classify_prompt = ChatPromptTemplate.from_messages([
    ("system", "Phân tích phản hồi khách hàng. Nhãn hợp lệ: khen, chê, hỏi, khác."),
    ("human", "{feedback}"),
])
structured_model = model.with_structured_output(FeedbackAnalysis)   # Pydantic model ở Ngày 2
analysis = (classify_prompt | structured_model).invoke({"feedback": "Áo mới mặc một lần đã bai cổ."})
print(analysis.label)   # đối tượng Pydantic, không phải chuỗi
```

---

## 6. invoke, stream, batch — khác nhau ở bản chất

### 6.1. invoke và stream: cùng tổng thời gian, khác trải nghiệm

Nhắc lại hai pha ở Ngày 1:

$$T_{\text{tổng}} \approx T_{\text{prefill}} + N_{\text{ra}} \times T_{\text{mỗi token}}$$

- `invoke`: người dùng chờ hết $T_{\text{tổng}}$ mới thấy chữ. Câu trả lời 400 token ở 48 token/giây là hơn 8 giây nhìn màn hình trống.
- `stream`: chữ đầu tiên xuất hiện sau khoảng $T_{\text{prefill}}$ (time to first token — thời gian tới token đầu tiên), phần còn lại chảy ra dần. **Tổng thời gian không đổi**, nhưng người dùng cảm thấy nhanh hơn hẳn.

```python
for piece in chain.stream({"text": "..."}):
    print(piece, end="", flush=True)
```

Streaming chỉ chảy qua được những bước **biết xử lý từng mảnh**:

- `StrOutputParser`, `JsonOutputParser`: xử lý được từng mảnh → vẫn stream.
- Một hàm thường bọc trong `RunnableLambda`: phải chờ đủ toàn bộ đầu vào → **chặn streaming** từ bước đó trở đi. Muốn stream qua hàm tự viết, viết hàm dạng generator nhận iterator đầu vào và `yield` từng mảnh.
- Trong chain 3 bước ở mục 3.1, bước 1 và 2 phải xong trọn vẹn thì bước 3 mới có đầu vào → chỉ bước cuối mới thật sự stream.

### 6.2. batch: song song, nhưng bị server giới hạn

`batch` chạy nhiều `invoke` song song bằng thread pool:

```python
results = chain.batch(
    list_of_inputs,
    config={"max_concurrency": 4},
    return_exceptions=True,
)
```

- `max_concurrency`: số request đồng thời tối đa. Đặt để không bị nhà cung cấp trả lỗi 429 (vượt giới hạn tần suất).
- `return_exceptions=True`: một đầu vào lỗi không làm hỏng cả lô; lỗi nằm đúng vị trí của nó trong danh sách kết quả.
- Thứ tự kết quả luôn khớp thứ tự đầu vào. Muốn nhận cái nào xong trước thì dùng `batch_as_completed`.

Lý tưởng: $n$ request, mỗi request mất thời gian $T$, chạy đồng thời $c$ request → khoảng $\lceil n / c \rceil \times T$. Thực tế **bị server chặn lại**:

- Ollama chỉ xử lý đồng thời tối đa `OLLAMA_NUM_PARALLEL` request cho mỗi model, phần còn lại xếp hàng. Các request song song còn chia nhau cùng một GPU 6 GB nên từng request chậm đi.
- API trả phí bị giới hạn số request và số token mỗi phút.

→ Trên máy local, `batch` có thể tăng **thông lượng tổng** một chút nhưng không nhân tuyến tính. Hãy đo — đây cũng là bài khởi động cho Tuần 9.

### 6.3. Cấu hình thêm cho Runnable

| Phương thức | Tác dụng | Tuần liên quan |
|---|---|---|
| `.bind(stop=["\n\n"])` | Gắn cố định tham số khi gọi model | Ngày 1 (stop sequence) |
| `.with_retry(stop_after_attempt=3)` | Tự thử lại khi gặp lỗi | Tuần 6, Tuần 13 |
| `.with_fallbacks([backup_model])` | Lỗi thì chuyển sang `Runnable` dự phòng | Tuần 13 |
| `.with_config(run_name="summarize", tags=["week_1"])` | Đặt tên, gắn nhãn cho tracing | Tuần 5 |

---

## 7. Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Báo thiếu biến khi `invoke` | Tên biến trong template khác khóa trong `dict` | In `prompt.input_variables` |
| Báo thiếu một biến lạ tên kiểu `"label"` | Dấu ngoặc nhọn của ví dụ JSON trong template | Nhân đôi thành `{{ }}` |
| Kết quả là `AIMessage(...)` thay vì chuỗi | Quên `StrOutputParser` ở cuối | Thêm parser |
| Kết quả có đoạn `<think>...</think>` | `qwen3:4b` đang bật thinking | `reasoning=False` trong `ChatOllama` (cần bản `langchain-ollama` mới) |
| `stream` không hiện chữ từ từ | Có `RunnableLambda` thường ở cuối chain | Chuyển hàm thành generator hoặc đưa nó ra khỏi chain |
| `batch` không nhanh hơn chạy tuần tự | Server xử lý tuần tự | Kiểm tra `OLLAMA_NUM_PARALLEL`, đo lại |

---

## 8. Câu hỏi tự kiểm tra

1. Trong `prompt | model | StrOutputParser()`, dữ liệu ở mỗi mũi tên có kiểu gì? Bỏ parser đi thì kết quả cuối có kiểu gì?
2. Vì sao một `dict` đặt trong chain lại chạy các nhánh song song?
3. Chain có một `RunnableLambda` dạng hàm thường đặt ở cuối. `stream` còn hiện chữ từ từ không?
4. `batch` 20 request tới Ollama với `max_concurrency = 8`, nhưng server chỉ xử lý 1 request mỗi lúc. So với gọi `invoke` tuần tự thì nhanh hơn bao nhiêu?
5. Khi nào dùng `with_structured_output` thay cho `JsonOutputParser`?

### Gợi ý đáp án

1. `dict` → `PromptValue` → `AIMessage` → `str`. Bỏ parser thì kết quả là `AIMessage`.
2. Khi ghép bằng `|`, `dict` được tự chuyển thành `RunnableParallel`, vốn chạy các nhánh trên cùng một đầu vào bằng thread pool.
3. Không. Hàm thường cần toàn bộ đầu vào nên phải chờ bước trước chạy xong, rồi trả kết quả một lần.
4. Gần như không nhanh hơn: 8 request được gửi cùng lúc nhưng server xếp hàng xử lý từng cái. Chỉ tiết kiệm được chút thời gian mạng giữa các request.
5. Khi model hoặc nhà cung cấp hỗ trợ structured output gốc (JSON schema hoặc tool calling), vì khi đó định dạng được đảm bảo bằng constrained decoding. `JsonOutputParser` hữu ích khi cần stream từng phần của JSON, hoặc model không hỗ trợ cơ chế gốc.

---

## Đọc thêm

- LangChain Docs — *Runnable interface*, *LCEL*, *Streaming*, *Structured output*.
- LangChain Docs — trang tích hợp *ChatOllama* và *OllamaEmbeddings*.
- [ollama_setup.md](../ollama_setup.md) mục 4.3 — ví dụ chain LangChain có streaming với `qwen3:4b`.
