# ai-translate-bot
### Tuần 1: Cơ bản về Python và AI

**Mục tiêu**: Hiểu Python và khái niệm chatbot cơ bản.

- **Ngày 1-2**: Học Python cơ bản
    - Nội dung: Biến, vòng lặp, hàm, thư viện (NumPy, Pandas).
    - Tài nguyên: [Python cơ bản trên freeCodeCamp](https://www.freecodecamp.org/learn/scientific-computing-with-python/) (miễn phí, ~5 giờ).
    - Thực hành: Viết script đơn giản (tính tổng 1 đến 10).
- **Ngày 3-4**: Làm quen với xử lý văn bản
    - Nội dung: Đọc file text/Excel, xử lý chuỗi (string).
    - Tài nguyên: [Python for Beginners - Learn Python in 1 Hour](https://www.youtube.com/watch?v=kqtD5dpn9C8) (YouTube, ~1 giờ).
    - Thực hành: Đọc file FAQ mẫu (ví dụ: “Câu hỏi: Giá bao nhiêu? Trả lời: 500k”).
- **Ngày 5-7**: Tìm hiểu chatbot là gì
    - Nội dung: Cách chatbot hoạt động (input → xử lý → output), khái niệm NLP.
    - Tài nguyên: [Intro to Chatbots - Hugging Face](https://huggingface.co/docs/transformers/tasks/conversational) (miễn phí).
    - Thực hành: Viết chatbot đơn giản (if-else) trả lời “Giá bao nhiêu?” bằng Python.
- **Thời gian**: ~20 giờ.

### Tuần 2-3: Làm quen với PyTorch và mô hình ngôn ngữ

**Mục tiêu**: Hiểu cách dùng thư viện AI và chạy mô hình cơ bản.

- **Ngày 8-10**: Cài đặt môi trường và học PyTorch
    - Nội dung: Cài Python, PyTorch trên máy tính; khái niệm tensor, gradient.
    - Tài nguyên: [PyTorch Tutorials](https://pytorch.org/tutorials/) (miễn phí, Beginner section).
    - Thực hành: Chạy ví dụ đơn giản (nhân ma trận) trên PyTorch.
- **Ngày 11-14**: Giới thiệu mô hình ngôn ngữ
    - Nội dung: Hiểu cách mô hình như BERT, GPT hoạt động; tải mô hình từ Hugging Face.
    - Tài nguyên: [Hugging Face Crash Course](https://www.youtube.com/watch?v=QEaBAZQC-tw) (~1 giờ).
    - Thực hành: Tải BERT-base, chạy thử trên Google Colab (miễn phí GPU) để sinh câu “Xin chào” bằng tiếng Việt.
- **Ngày 15-21**: Thử nghiệm chatbot cơ bản
    - Nội dung: Dùng pipeline của Hugging Face để tạo chatbot đơn giản.
    - Tài nguyên: [Conversational Pipeline](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial).
    - Thực hành: Chạy mô hình DialoGPT trên Colab, hỏi đáp vài câu tiếng Việt.
- **Thời gian**: ~40-50 giờ.

### Tuần 4-5: Fine-Tuning và RAG cơ bản

**Mục tiêu**: Học cách điều chỉnh mô hình và kết hợp truy xuất thông tin.

- **Ngày 22-25**: Fine-Tuning mô hình nhỏ
    - Nội dung: Hiểu Fine-Tuning, cách dùng dataset để huấn luyện.
    - Tài nguyên: [Fine-Tuning Tutorial - Hugging Face](https://huggingface.co/docs/transformers/training).
    - Thực hành: Fine-Tune DistilBERT trên Colab với dataset FAQ nhỏ (10 câu hỏi - trả lời).
- **Ngày 26-30**: Tìm hiểu RAG
    - Nội dung: Cách kết hợp truy xuất (retrieval) và sinh câu (generation).
    - Tài nguyên: [LangChain RAG Tutorial](https://python.langchain.com/docs/get_started/introduction) (miễn phí).
    - Thực hành: Dùng LangChain + FAISS trên Colab để tạo chatbot trả lời từ file text (như danh sách sản phẩm).
- **Ngày 31-35**: Kết hợp FT và RAG
    - Nội dung: Tích hợp mô hình Fine-Tuned với RAG.
    - Thực hành: Xây chatbot trả lời “Giá áo đỏ bao nhiêu?” dựa trên file Excel mẫu.
- **Thời gian**: ~40-50 giờ.

### Tuần 6-7: Thu thập dữ liệu và xây dự án thực tế

**Mục tiêu**: Có chatbot demo và dữ liệu sẵn sàng cho Digits.

- **Ngày 36-39**: Thu thập dữ liệu tiếng Việt
    - Nội dung: Dùng data set có sẵn hoặc đi craw data báo trên mạng
    - Tài nguyên: [BeautifulSoup Tutorial](https://www.youtube.com/watch?v=87Gx3U0BDlo) (~1 giờ).
    - Thực hành: Crawl 100 bài báo tiếng Việt, lưu thành file text.
- **Ngày 40-44**: Làm sạch và chuẩn bị dữ liệu
    - Nội dung: Xử lý dữ liệu (loại bỏ nhiễu, chuẩn hóa tiếng Việt).
    - Thực hành: Chuyển dữ liệu thô thành định dạng huấn luyện (câu hỏi - trả lời).
- **Ngày 45-49**: Xây chatbot demo
    - Nội dung: Tạo chatbot hoàn chỉnh trên Colab.
    - Thực hành: Chatbot trả lời câu hỏi về sản phẩm (ví dụ: “Áo này còn không?” → “Còn 2 cái!”).
- **Thời gian**: ~40 giờ.

### Tuần 8: Triển khai và chuẩn bị bàn giao

**Mục tiêu**: Có sản phẩm chạy được và kế hoạch dùng Digits.

- **Ngày 50-52**: Tích hợp giao diện
    - Nội dung: Chuyển chatbot sang giao diện đơn giản (terminal hoặc web).
    - Tài nguyên: React hoặc NextJS hoặc dùng template cho AI có sẵn
- **Ngày 53-55**: Kiểm tra và tối ưu
    - Nội dung: Test chatbot với 20-30 câu hỏi, sửa lỗi.
    - Thực hành: Đảm bảo chatbot trả lời đúng 80% trường hợp.
- **Ngày 56**: Tìm cách training với data set lớn hơn với model lớn hơn
    - Nội dung: Chuẩn bị Fine-Tune mô hình lớn (LLaMA 13B) với dữ liệu đã có,
    - Phân cứng: Project Digits
    - Thực hành: Viết kế hoạch (cài PyTorch, tải LLaMA, chạy FT trên Digits).
- **Thời gian**: ~20 giờ.