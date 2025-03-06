from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch
# Kiểm tra và chọn device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load model và tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load dataset từ Hugging Face
ds = load_dataset("Helsinki-NLP/opus-100", "en-vi")
dataset = ds["train"]

# Chuyển đổi dữ liệu cho nhiệm vụ QA
def preprocess_function(examples):
    inputs = ["translate to Vietnamese: " + ex["en"] for ex in examples["translation"]]
    targets = [ex["vi"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Thiết lập tham số training
training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_t5_small",
    evaluation_strategy="no",
    learning_rate=5e-4,
    per_device_train_batch_size=8,  # Tăng batch size nếu có GPU
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
)

# Tạo data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Tạo trainer và fine-tune
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# Lưu model
model.save_pretrained("./fine_tuned_t5_small")
tokenizer.save_pretrained("./fine_tuned_t5_small")
print("Fine-tuning hoàn tất! Model đã được lưu vào './fine_tuned_t5_small'")