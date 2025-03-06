from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model và tokenizer
model_name = "google-t5/t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
# Hàm để tạo văn bản
def generate_text(input_text):
    # Thêm tiền tố nhiệm vụ
    input_text = "translate to Vietnamese: " + input_text
    
    # Mã hóa input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Tạo output
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,
        num_beams=4,
        early_stopping=True
    )
    
    # Giải mã và trả về kết quả
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test với một ví dụ
test_input = "The quick brown fox jumps over the lazy dog."
print("Trước khi fine-tuning:")
print("Input:", test_input)
print("Output:", generate_text(test_input))