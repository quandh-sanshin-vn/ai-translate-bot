from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model đã fine-tune
model_path = "./fine_tuned_t5_small"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def generate_text(input_text):
    input_text = "translate to Vietnamese: " + input_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
test_input = "The quick brown fox jumps over the lazy dog."
print("Sau khi fine-tuning:")
print("Input:", test_input)
print("Output:", generate_text(test_input))