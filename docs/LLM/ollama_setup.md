# Chạy LLM local với Ollama

Hướng dẫn tự cài Ollama trên máy cá nhân (Windows 11) và chọn model vừa với phần cứng.
Đây là bước **Tuần 0** trong [learning_plan.md](learning_plan.md). Model local dùng để luyện tập hằng ngày mà không tốn tiền API.

---

## 1. Cấu hình máy & ngân sách VRAM

| Thành phần | Thông số | Ảnh hưởng |
|---|---|---|
| Máy | Dell G15 5520 (laptop) | Phải **cắm sạc** khi chạy model, chạy pin GPU sẽ bị hạ xung |
| CPU | Intel i7-12700H, 14 nhân / 20 luồng | Đủ khỏe để gánh phần model bị tràn ra khỏi GPU, nhưng chậm hơn GPU nhiều lần |
| RAM | 16 GB | Không chạy nổi model ≥ 14B một cách thoải mái |
| GPU | NVIDIA RTX 3060 Laptop, **6 GB VRAM**, compute 8.6 | Đây là giới hạn quyết định |
| GPU tích hợp | Intel Iris Xe | Ollama không dùng, bỏ qua |
| Ổ đĩa | C: còn ~34 GB, D: còn ~176 GB | **Bắt buộc để model ở ổ D:** |

Windows và các app đang mở đã chiếm khoảng 1 GB VRAM, nên thực tế còn **~5 GB** cho model.

### Quy tắc tính nhanh

```text
VRAM cần ≈ dung lượng file model + KV cache + ~0.5 GB overhead
```

KV cache tăng tuyến tính theo độ dài context. Ví dụ với `qwen3:4b` (36 layer, 8 KV head, head_dim 128):

```text
KV / token = 2 (K,V) × 36 layer × 8 head × 128 dim × 2 byte (FP16) ≈ 144 KB
Context 8K  → ~1.1 GB (FP16)  hoặc ~0.6 GB nếu nén KV cache về q8_0
Context 16K → ~2.3 GB (FP16)  hoặc ~1.1 GB (q8_0)
```

| Cấu hình `qwen3:4b` | Tổng VRAM ước tính | Chạy 100% GPU? |
|---|---|---|
| 8K context, KV FP16 | 2.5 + 1.1 + 0.5 ≈ **4.1 GB** | ✅ |
| 16K context, KV q8_0 | 2.5 + 1.1 + 0.5 ≈ **4.1 GB** | ✅ |
| 16K context, KV FP16 | 2.5 + 2.3 + 0.5 ≈ **5.3 GB** | ⚠️ sát giới hạn, dễ tràn sang CPU |

> Đây chính là bài tập tính VRAM của **Tuần 9 – D1**. Làm quen luôn từ bây giờ.

---

## 2. Model đề xuất cho máy này

| Vai trò | Model | Dung lượng | Vừa 6 GB VRAM? | Dùng cho |
|---|---|---|---|---|
| ⭐ **Chat / agent chính** | `qwen3:4b` | ~2.5 GB | ✅ 100% GPU | Hằng ngày, RAG, **tool calling** (Tuần 6–8) |
| Chất lượng cao hơn | `qwen3:8b` | ~5.2 GB | ⚠️ tràn một phần sang CPU | So sánh chất lượng, chấp nhận chậm |
| Tiếng Việt / ảnh | `gemma3:4b` | ~3.3 GB | ✅ | Viết tiếng Việt tự nhiên, hỏi đáp trên ảnh |
| ⭐ **Embedding** | `bge-m3` | ~1.2 GB | ✅ | Đa ngữ, 1024 chiều, context 8K token |
| Siêu nhẹ cho test | `qwen3:1.7b` | ~1.4 GB | ✅ | Unit test, CI, thử pipeline cho nhanh |

### Vì sao chọn những model này

- **`qwen3:4b`** — vừa hoàn toàn trong GPU kể cả khi context 8–16K, hỗ trợ **tool calling** (bắt buộc cho phần agentic), có chế độ *thinking* bật/tắt được, hỗ trợ tiếng Việt tốt. Đây là lựa chọn cân bằng nhất cho máy 6 GB.
- **`bge-m3`** — embedding đa ngữ, khớp với lựa chọn ở Tuần 1 – D4. Lưu ý: qua Ollama chỉ lấy được **dense vector**, còn phần sparse/multi-vector của BGE-M3 thì không. Phần BM25 cho hybrid search ở Tuần 3 phải làm riêng.
- **`gemma3:4b`** — viết tiếng Việt khá tự nhiên và **nhận được ảnh đầu vào**, hợp để tận dụng nền CV của bạn. Nhưng trên Ollama nó không hỗ trợ tool calling, nên không dùng cho các tuần agent.

### Không nên dùng trên máy này

| Model | Lý do |
|---|---|
| `qwen3:14b`, `gemma3:12b` trở lên | Phần lớn model nằm trên CPU → chậm tới mức mất hứng học |
| `gpt-oss:20b` | Cần ~14 GB bộ nhớ, sẽ chiếm gần hết 16 GB RAM của máy |
| `llama3.1:8b`, `llama3.2` | Tiếng Việt không nằm trong danh sách ngôn ngữ được hỗ trợ chính thức; bản 8B còn không vừa VRAM |

!!! tip "Khi có model mới"
    Model ra liên tục, danh sách trên sẽ cũ đi. Khi xem một model mới trên [ollama.com/library](https://ollama.com/library), áp dụng đúng quy tắc: **dung lượng file + 1–1.5 GB ≤ 5 GB** thì chạy được 100% trên GPU.
    Cần tool calling thì xem model có tag **tools** hay không.

---

## 3. Cài đặt trên Windows

### Bước 0 — Chuẩn bị

- Cắm sạc, chọn chế độ nguồn *Best performance*.
- Tắt bớt các app ăn VRAM (game, trình duyệt nhiều tab video, app có bật hardware acceleration).
- Driver NVIDIA hiện tại (581.95) đã đủ mới, không cần cập nhật.

### Bước 1 — Chuyển thư mục model sang ổ D: (làm TRƯỚC khi tải model)

Mặc định Ollama lưu model ở `C:\Users\<user>\.ollama\models`. Ổ C: chỉ còn ~34 GB nên phải đổi ngay:

```powershell
New-Item -ItemType Directory -Force D:\ollama\models
[Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\ollama\models", "User")
```

### Bước 2 — Cài Ollama

Chọn một trong hai cách:

```powershell
# Cách 1: winget
winget install --id Ollama.Ollama -e
```

Cách 2: tải `OllamaSetup.exe` từ [ollama.com/download](https://ollama.com/download).
Nếu muốn cài luôn phần mềm vào ổ D: (bản thân app kèm thư viện CUDA cũng tốn vài GB):

```powershell
.\OllamaSetup.exe /DIR="D:\Programs\Ollama"
```

Sau khi cài, Ollama tự chạy nền dưới khay hệ thống (system tray) và lắng nghe tại `http://localhost:11434`.
**Không cần chạy `ollama serve`** nữa — chạy thêm sẽ báo lỗi cổng 11434 đã bị chiếm.

### Bước 3 — Biến môi trường tối ưu cho máy 6 GB VRAM

```powershell
# Bật flash attention (điều kiện bắt buộc để nén KV cache)
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
# Nén KV cache về 8-bit: giảm ~một nửa VRAM cho context, chất lượng gần như không đổi
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
# Đặt context mặc định rõ ràng, tránh prompt RAG bị cắt âm thầm
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH", "8192", "User")
```

Sau đó **thoát Ollama** (chuột phải icon dưới khay → *Quit Ollama*) rồi mở lại để nó nhận biến môi trường mới.

Kiểm tra lại:

```powershell
[Environment]::GetEnvironmentVariables("User").GetEnumerator() | Where-Object Name -like "OLLAMA_*"
```

!!! note "Về trang Settings của app"
    Các bản Ollama mới có trang Settings trong app, cho phép chỉnh vị trí lưu model và độ dài context bằng giao diện. Nếu bản của bạn có thì dùng cách đó cũng được, nhưng biến môi trường là cách chắc chắn nhất và dùng được cho mọi phiên bản.

### Bước 4 — Tải và chạy model

```powershell
ollama pull qwen3:4b
ollama pull bge-m3
ollama run qwen3:4b
```

Gõ thử một câu tiếng Việt, gõ `/bye` để thoát.

### Bước 5 — Kiểm tra model có thực sự chạy trên GPU

Trong lúc model đang được load (trong vòng 5 phút sau lần gọi cuối), chạy:

```powershell
ollama ps
```

```text
NAME        ID              SIZE      PROCESSOR    UNTIL
qwen3:4b    ...             4.0 GB    100% GPU     4 minutes from now
```

- `100% GPU` → tốt.
- Kiểu `35%/65% CPU/GPU` → model bị tràn sang CPU. Giảm context, đóng app ăn VRAM, hoặc dùng model nhỏ hơn.

Đo tốc độ và **ghi con số này vào [progress_tracker.md](progress_tracker.md)** — nó sẽ là baseline cho phần benchmark ở Tuần 9:

```powershell
ollama run qwen3:4b --verbose "Giải thích RAG trong 3 câu."
```

Dòng cần chú ý ở cuối output là `eval rate: ... tokens/s` (tốc độ sinh token) và `prompt eval rate` (tốc độ xử lý prompt).

---

## 4. Sử dụng

### 4.1. Lệnh CLI hay dùng

| Lệnh | Tác dụng |
|---|---|
| `ollama list` | Liệt kê model đã tải |
| `ollama ps` | Model đang nằm trong bộ nhớ + đang chạy trên GPU hay CPU |
| `ollama show qwen3:4b` | Xem thông tin: kiến trúc, context tối đa, khả năng (tools, vision, thinking) |
| `ollama stop qwen3:4b` | Giải phóng VRAM ngay thay vì chờ 5 phút |
| `ollama rm <model>` | Xóa model để lấy lại dung lượng |

Trong màn hình chat của `ollama run`:

| Lệnh | Tác dụng |
|---|---|
| `/set nothink` | Tắt chế độ thinking của Qwen3 (trả lời ngay, không in phần suy luận dài) |
| `/set parameter num_ctx 16384` | Đổi độ dài context cho phiên hiện tại |
| `/show info` | Xem thông tin model |
| `/bye` | Thoát |

### 4.2. Gọi REST API

Ollama có sẵn API native và API tương thích OpenAI. Trong PowerShell dùng `Invoke-RestMethod` (lệnh `curl` trong PowerShell 5.1 là alias của lệnh khác, dễ gây nhầm):

```powershell
# Chat
$body = @{
  model    = "qwen3:4b"
  messages = @(@{ role = "user"; content = "Xin chào, bạn là ai?" })
  stream   = $false
  think    = $false
} | ConvertTo-Json -Depth 5
(Invoke-RestMethod http://localhost:11434/api/chat -Method Post -Body $body -ContentType "application/json").message.content

# Embedding
$body = @{ model = "bge-m3"; input = @("Hà Nội là thủ đô của Việt Nam") } | ConvertTo-Json
(Invoke-RestMethod http://localhost:11434/api/embed -Method Post -Body $body -ContentType "application/json").embeddings[0].Count
# → 1024
```

### 4.3. Python

Cài vào môi trường conda đang dùng cho kế hoạch học:

```powershell
conda create -n llm-lab python=3.11 -y   # bỏ qua nếu đã có môi trường
conda activate llm-lab
pip install ollama langchain-ollama openai
```

**Client chính thức của Ollama:**

```python
import ollama

resp = ollama.chat(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Tóm tắt RAG trong 2 câu."}],
    think=False,                       # tắt thinking của Qwen3
    options={"temperature": 0.2, "num_ctx": 8192},
)
print(resp.message.content)

emb = ollama.embed(model="bge-m3", input=["câu thứ nhất", "câu thứ hai"])
print(len(emb.embeddings), len(emb.embeddings[0]))   # 2 1024
```

**Structured output** (bài tập Tuần 1 – D2): truyền JSON schema vào `format`, model buộc phải trả JSON đúng schema.

```python
from pydantic import BaseModel
import ollama

class Answer(BaseModel):
    summary: str
    keywords: list[str]

resp = ollama.chat(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Phân tích: RAG giúp LLM trả lời dựa trên tài liệu nội bộ."}],
    format=Answer.model_json_schema(),
    think=False,
)
print(Answer.model_validate_json(resp.message.content))
```

**Endpoint tương thích OpenAI** — nên tập dùng cách này, vì code viết theo chuẩn OpenAI sẽ chuyển được sang NVIDIA NIM (Tuần 10) hay vLLM (Tuần 9) chỉ bằng cách đổi `base_url`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key bắt buộc có nhưng không được kiểm tra
resp = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Xin chào"}],
)
print(resp.choices[0].message.content)
```

**LangChain** (Tuần 1 – D3 trở đi):

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="qwen3:4b", temperature=0, num_ctx=8192, reasoning=False)
embeddings = OllamaEmbeddings(model="bge-m3")

chain = ChatPromptTemplate.from_template("Tóm tắt đoạn sau trong 1 câu:\n{text}") | llm
for chunk in chain.stream({"text": "Ollama giúp chạy LLM trên máy cá nhân..."}):
    print(chunk.content, end="", flush=True)
```

> Tham số `reasoning` chỉ có ở các bản `langchain-ollama` mới. Nếu báo lỗi, hãy nâng cấp package hoặc bỏ tham số đó đi.

### 4.4. Tạo model riêng bằng Modelfile

Gói sẵn system prompt và tham số để khỏi lặp lại trong code. Tạo file `Modelfile`:

```text
FROM qwen3:4b
PARAMETER num_ctx 8192
PARAMETER temperature 0.2
SYSTEM """Bạn là trợ lý kỹ thuật. Luôn trả lời bằng tiếng Việt, ngắn gọn.
Nếu không chắc chắn thì nói "không biết", không được bịa."""
```

```powershell
ollama create qwen3-vi -f Modelfile
ollama run qwen3-vi
```

---

## 5. Kết nối từ Docker (dùng từ Tuần 11)

Khi API service chạy trong container còn Ollama chạy trên Windows, container gọi Ollama qua:

```text
http://host.docker.internal:11434
```

!!! warning "Bảo mật"
    Ollama **không có cơ chế xác thực**. Đừng đặt `OLLAMA_HOST=0.0.0.0` để mở ra mạng LAN khi đang dùng Wi-Fi công cộng, vì ai cùng mạng cũng gọi được model của bạn.
    Đây cũng là ví dụ thực tế đầu tiên cho phần authN/authZ ở Tuần 14.

---

## 6. Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Sinh chữ rất chậm | Model tràn sang CPU | Xem `ollama ps`. Giảm `num_ctx`, bật KV `q8_0`, đóng app dùng GPU, cắm sạc, hoặc dùng model nhỏ hơn |
| Ổ C: vẫn bị đầy | Ollama chưa nhận `OLLAMA_MODELS` | Thoát hẳn Ollama ở khay hệ thống rồi mở lại. Model đã tải trước đó thì chuyển thủ công từ `%USERPROFILE%\.ollama\models` sang `D:\ollama\models` |
| RAG trả lời như thể không đọc context | Context quá nhỏ, prompt bị **cắt âm thầm** | Đặt `num_ctx` đủ lớn cho cả prompt + các chunk. Đếm token trước khi gửi |
| Qwen3 in ra đoạn suy luận rất dài trước câu trả lời | Chế độ thinking đang bật | `think=False` (API/Python), `/set nothink` (CLI) |
| `bind: Only one usage of each socket address` | Ollama đã chạy sẵn dưới khay | Không cần `ollama serve`, cứ dùng luôn |
| Chat model và embedding model thay nhau load lại, rất chậm | Không đủ VRAM để giữ cả hai cùng lúc | Khi ingest hàng loạt, chạy embedding trên CPU bằng `options={"num_gpu": 0}` để nhường GPU cho chat model |
| Tiếng Việt hiển thị lỗi trong terminal | Console cũ không xử lý tốt UTF-8 | Dùng Windows Terminal |

---

## 7. Checklist hoàn thành

- [ ] `OLLAMA_MODELS` trỏ về `D:\ollama\models`, model tải về nằm đúng ở đó
- [ ] `qwen3:4b` chạy với `ollama ps` hiển thị **100% GPU**
- [ ] Đã ghi `eval rate` (tokens/s) vào progress tracker
- [ ] `bge-m3` trả về vector 1024 chiều
- [ ] Gọi được model từ Python qua cả client `ollama` lẫn endpoint OpenAI-compatible
- [ ] Chạy được một LangChain chain có streaming

## Tham khảo

- [Ollama docs](https://docs.ollama.com/) — đặc biệt các mục *FAQ*, *Windows*, *API*
- [Ollama model library](https://ollama.com/library)
- [Ollama Python client](https://github.com/ollama/ollama-python)
- [LangChain – Ollama integration](https://python.langchain.com/docs/integrations/chat/ollama/)
