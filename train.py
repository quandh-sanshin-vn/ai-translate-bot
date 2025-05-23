from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from button_dataset import data

# 1. Chuẩn bị dữ liệu
df = pd.DataFrame(data)
df = df.dropna(subset=['JP', 'VI'])
df['JP'] = df['JP'].str.strip()
df['VI'] = df['VI'].str.strip()

print("Dữ liệu mẫu:")
print(df.head())

dataset = Dataset.from_pandas(df[['JP', 'VI']])

# 2. Tải mô hình và tokenizer
model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Ngôn ngữ nguồn và đích
src_lang = "ja"
tgt_lang = "vi"
tokenizer.src_lang = src_lang
tokenizer.tgt_lang = tgt_lang

# 3. Cấu hình LoRA
config = LoraConfig(
    r=8,  # Rank của LoRA
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Chỉ áp dụng LoRA cho các module này
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

# Chuẩn bị mô hình cho LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)

# 4. Tokenize dữ liệu
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

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Cấu hình huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-4,  # Learning rate cao hơn một chút vì chỉ fine-tune tham số LoRA
    warmup_steps=500,
    save_total_limit=2,
    report_to="tensorboard",
)

# 6. Huấn luyện mô hình
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 7. Lưu lại trọng số LoRA
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

print("Hoàn thành fine-tune với LoRA và lưu mô hình tại ./results")
