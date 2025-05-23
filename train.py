from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import pandas as pd
from transformers import Trainer, TrainingArguments

# Import dữ liệu từ dataset.py
from data import data

# Chuyển dữ liệu thành pandas DataFrame
df = pd.DataFrame(data)

# Xử lý dữ liệu (nếu cần)
df = df.dropna(subset=['JP', 'VI'])
df['JP'] = df['JP'].str.strip()
df['VI'] = df['VI'].str.strip()

# Kiểm tra dữ liệu đầu ra
print("Dữ liệu mẫu:")
print(df.head())

# Tải tokenizer và mô hình
model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Cấu hình ngôn ngữ
src_lang = "ja"  # Tiếng Nhật
tgt_lang = "vi"  # Tiếng Việt
tokenizer.src_lang = src_lang
tokenizer.tgt_lang = tgt_lang

# Chuyển đổi dữ liệu pandas DataFrame thành Hugging Face Dataset
dataset = Dataset.from_pandas(df[['JP', 'VI']])

# Hàm tokenization
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples['JP'], truncation=True, padding="max_length", max_length=128
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['VI'], truncation=True, padding="max_length", max_length=128
        )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Áp dụng tokenization lên dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Thiết lập tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    bf16=True,
)
model.gradient_checkpointing_enable()
# Tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Huấn luyện mô hình
trainer.train()

# Lưu mô hình và tokenizer sau khi huấn luyện
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

print("Hoàn thành fine-tune và lưu mô hình tại ./results")
