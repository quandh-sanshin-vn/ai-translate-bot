from transformers import AutoTokenizer, AutoModelForCausalLM
print("Đang tải tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
print("Đang tải mô hình Mistral...")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
print("Tải hoàn tất!")