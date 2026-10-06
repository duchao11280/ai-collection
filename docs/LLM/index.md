# LLM

!!! tip "Bắt đầu từ đâu"
    - **[Kế hoạch học LLM hằng ngày (16 tuần)](learning_plan.md)** — lộ trình chi tiết theo từng ngày.
    - [Tuần 1 — Lý thuyết nền tảng](week_01/index.md) — token, sampling, prompt, LangChain, embedding, Chroma.
    - [Tài liệu học theo chủ đề](resources.md)
    - [Bảng theo dõi tiến độ](progress_tracker.md)
    - [Chạy LLM local với Ollama](ollama_setup.md)

Phần dưới đây là ghi chú nền tảng về RAG, retrieval agent và LangChain.

---

# RAG
Retrieval-Augmented Generation (RAG) là một framework AI kết hợp khả năng của các LLMs với cơ sở kiến thức từ bên ngoài nhằm tạo ra các phản hồi chính xác và phù hợp mà không cần phải training lại hoặc phải fine-tuning model.

LLM nền tảng thường được huấn luyện trên dữ liệu phổ quát và có khả năng xử lý ngôn ngữ tự nhiên tốt.
Tuy nhiên, chúng chỉ hiểu biết những gì đã được học trong quá trình huấn luyện, dẫn đến hạn chế trong việc xử lý các câu hỏi yêu cầu thông tin cập nhật hoặc chuyên biệt.
RAG bổ sung cho LLMs bằng cách kết nối với các nguồn dữ liệu bên ngoài (như cơ sở dữ liệu nội bộ của tổ chức).
Khi có truy vấn, RAG truy xuất dữ liệu từ các nguồn này, sau đó kết hợp thông tin đó với khả năng xử lý ngôn ngữ của LLM để tạo ra câu trả lời chính xác và có ngữ cảnh.

RAG ra đời để khắc phục việc LLM đưa ra các thông tin sai lệch khi không có câu trả lời, thông tin bị lỗi thời, vấn đề hallucination 
(sinh ra các câu văn nghe hợp lý nhưng không chính xác, hay gọi là “ảo giác”)

## Quy trình hoạt động của RAG

1. Xây dựng cơ sở tri thức (Knowledge Base): Chuẩn bị phần dữ liệu như văn bản nội bộ, sổ tay, tài liệu sản phẩm, báo cáo,... Loại bỏ các dữ liệu trùng lặp hoặc không cần thiết. Chia nhỏ dữ liệu thành các phần để dễ xử lý (chunk).
2. Chuyển đổi dữ liệu thành vector: Sử dụng embedding model để chuyển đổi văn bản thành các vector. Các vector này lưu trữ ngữ cảnh và các mối liên hệ giữa các từ. Và các vector được lưu trữ trong vector database để có thể dễ truy xuất.
3. Truy xuất thông tin: Khi có truy vấn từ người dùng hoặc hệ thống, truy vấn sẽ được gửi tới thành phần cốt lõi của hệ thống (cơ chế truy xuất) để làm việc. Cơ chế này tìm kiếm trong vector database để tìm ra các dữ liệu liên quan nhất. Các thông tin phù hợp được gửi đến LLM dưới dạng ngữ cảnh bổ sung.
4. Tạo phản hồi: LLM sử dụng thông tin truy xuất được từ knowledge base cùng với khả năng xử lý ngôn ngữ tự nhiên của nó để tạo ra câu trả lời phù hợp.


# Dialog/Retrieval Agents

## LLM orchestration
LLM Orchestration: Software + LLM helps to route to software and LLMs

## Retrieval
Retrieval: Tool runs algorithms (database, code excution, semantic search, return a constant value, ...) to provide context.

## Augmented
Augmented: Based on tool responses, the software pipeline synthesizes some "context" to feed to LLM w/question.

## Generation
Generation: Based on question, instructions, and  enhanced context, the LLM returns a response.

# LangChain

LangChain connector chấp nhận input và tạo output từ LLM. Hỗ trợ thêm các thành phần bổ sung vào input như thông điệp hệ thống (system messages) hoặc các thành phần khác.
Runnables là một cách tiếp cận trong LangChain cho phép kết hợp các module chức năng thành các pipeline.
Có thể dễ dàng định nghĩa các Runnables bằng cách tạo một hàm nhận đầu vào từ bước trước và tạo đầu ra cho bước tiếp theo. Sau đó, thêm nó vào chuỗi (chain).

Triển khai chuỗi (chain):
Một chuỗi có thể được sử dụng để: Gọi một lần để nhận toàn bộ phản hồi; hoặc stream từng token một.

# Ref

[What is RAG?](https://www.intel.com/content/www/us/en/learn/what-is-rag.html)