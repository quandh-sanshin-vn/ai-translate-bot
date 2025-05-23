from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import pandas as pd
from transformers import Trainer, TrainingArguments

from data import data

df = pd.DataFrame(data)

df = df.dropna(subset=['JP', 'VI'])
df['JP'] = df['JP'].str.strip()
df['VI'] = df['VI'].str.strip()

print("Dữ liệu mẫu:")
print(df.head())

model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

src_lang = "ja"
tgt_lang = "vi"
tokenizer.src_lang = src_lang
tokenizer.tgt_lang = tgt_lang

dataset = Dataset.from_pandas(df[['JP', 'VI']])

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

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="steps",
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to="tensorboard",
)
model.gradient_checkpointing_enable()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

print("Hoàn thành fine-tune và lưu mô hình tại ./results")
