import gspread
from oauth2client.service_account import ServiceAccountCredentials
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import pandas as pd
from transformers import Trainer, TrainingArguments

# Xác thực với Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('test-training-data-245841565d66.json', scope)
client = gspread.authorize(creds)

# Mở Google Sheet và lấy dữ liệu
sheet = client.open('DATA_TRAINING_AI_TRANSLATE').sheet1
data = sheet.get_all_records()

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Xử lý dữ liệu (loại bỏ NaN, cắt chuỗi)
df = df.dropna(subset=['Source', 'Target'])
df['Source'] = df['Source'].str.strip()
df['Target'] = df['Target'].str.strip()

# In ra một số dòng dữ liệu đầu tiên để kiểm tra
print(df.head())

# Tải tokenizer và mô hình NLLB (facebook/nllb-200-distilled-600M)
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer.src_lang = "ja"  # Ngôn ngữ nguồn là Tiếng Nhật
tokenizer.tgt_lang = "vi"  # Ngôn ngữ đích là Tiếng Việt

# Chuyển đổi dữ liệu pandas DataFrame thành Hugging Face Dataset
train_dataset = Dataset.from_pandas(df[['Source', 'Target']])

# Hàm tokenization cho dữ liệu
def tokenize_function(examples):
    if not examples["Source"] or not examples["Target"]:
        return {}
    model_inputs = tokenizer(examples["Source"], truncation=True, padding="max_length", max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["Target"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Áp dụng tokenization lên dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)

# Thiết lập các tham số huấn luyện
training_args = TrainingArguments(
    output_dir='./results',                # Đường dẫn lưu kết quả
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch",                 # Lưu mô hình sau mỗi epoch
    logging_dir='./logs',
    logging_steps=10
)

# Huấn luyện mô hình
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Lưu mô hình và tokenizer sau khi huấn luyện
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
