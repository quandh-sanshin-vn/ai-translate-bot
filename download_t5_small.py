from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print("Đang tải tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
print("Đang tải mô hình Mistral...")
model =  AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
print("Tải hoàn tất!")