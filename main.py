from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gspread
from google.oauth2.service_account import Credentials
import datetime
import time

# Flask app
app = Flask(__name__)
# Google Sheets authentication
CREDENTIALS_FILE = 'test-training-data-245841565d66.json'
SPREADSHEET_ID = '1spg80mi-dbMZX_97DJrv2PQnA_f7pr1Elprvn3CP9WU'
SHEET_NAME = 'LOG_COMMON'
MODEL_NAME = "facebook/nllb-200-1.3B"
SPECIAL_TERMS_TO_ENGLISH = {
    "チェックボックス": "checkbox",
    "ボタン": "button"
}



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


# Hàm xác thực Google Sheets
def authenticate_google_sheets():
    creds = Credentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    client = gspread.authorize(creds)
    return client


def load_special_terms_from_sheet():
    # Xác thực Google Sheets
    client = authenticate_google_sheets()
    sheet = client.open("DATA_TRAINING_AI_TRANSLATE").worksheet("SPECIAL_TERMS_TO_ENGLISH")
    data = sheet.get_all_values()

    # Chuyển dữ liệu thành dictionary
    special_terms_to_english = {row[0]: row[1] for row in data if len(row) >= 2}  # Cột A -> key, Cột B -> value
    return special_terms_to_english


# Ghi log vào Google Sheets
def write_to_google_sheet(sheet_name, source_text, target_text, source_lang, target_lang, current_device, inference_time):
    client = authenticate_google_sheets()
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(sheet_name)

    log_data = [
        str(datetime.datetime.now()),  # Thời gian
        source_text,  # Văn bản nguồn
        target_text,  # Văn bản đích
        source_lang,  # Ngôn ngữ nguồn
        target_lang,  # Ngôn ngữ đích
        current_device,  # Thiết bị hiện tại (GPU/CPU)
        f"{inference_time:.2f}"  # Thời gian xử lý
    ]
    sheet.append_row(log_data)


@app.route('/translate_no_sheet_name', methods=['POST'])
def translate_no_sheet_name():
    start_time = time.time()
    data = request.get_json()
    article = data.get('text')  # Văn bản gốc từ API
    source_lang = data.get('source_lang', 'ja')
    target_lang = data.get('target_lang', 'vi')

    current_device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Current Device: {current_device}")

    replaced_terms = {}
    for term, english_term in SPECIAL_TERMS_TO_ENGLISH.items():
        placeholder = f"[SPECIAL_TERM_{len(replaced_terms)}]"
        article = article.replace(term, placeholder)
        replaced_terms[placeholder] = english_term

    if source_lang == 'ja' and target_lang == 'vi':
        tokenizer.src_lang = "jpn_Jpan"
        tokenizer.tgt_lang = "vie_Latn"
    elif source_lang == 'vi' and target_lang == 'ja':
        tokenizer.src_lang = "vie_Latn"
        tokenizer.tgt_lang = "jpn_Jpan"

    inputs = tokenizer(article, return_tensors="pt", padding=True).to(device)
    max_length = 1000

    if target_lang == 'vi':
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("vie_Latn")
    elif target_lang == 'ja':
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("jpn_Jpan")
    else:
        return jsonify({'error': 'Unsupported target language'}), 400

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    for placeholder, english_term in replaced_terms.items():
        translated_text = translated_text.replace(placeholder, english_term)

    end_time = time.time()
    inference_time = end_time - start_time
    write_to_google_sheet(SHEET_NAME, data.get('text'), translated_text, source_lang, target_lang, current_device, inference_time)

    print(f"Inference Time: {inference_time:.2f} seconds")
    return jsonify({'translated_text': translated_text})

@app.route('/translate', methods=['POST'])
def translate():
    start_time = time.time()
    data = request.get_json()
    article = data.get('text')  # Văn bản gốc từ API
    source_lang = data.get('source_lang', 'ja')
    target_lang = data.get('target_lang', 'vi')
    sheet_name = data.get('sheet_name', SHEET_NAME)

    current_device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Current Device: {current_device}")

    replaced_terms = {}
    for term, english_term in SPECIAL_TERMS_TO_ENGLISH.items():
        placeholder = f"[SPECIAL_TERM_{len(replaced_terms)}]"
        article = article.replace(term, placeholder)
        replaced_terms[placeholder] = english_term

    if source_lang == 'ja' and target_lang == 'vi':
        tokenizer.src_lang = "jpn_Jpan"
        tokenizer.tgt_lang = "vie_Latn"
    elif source_lang == 'vi' and target_lang == 'ja':
        tokenizer.src_lang = "vie_Latn"
        tokenizer.tgt_lang = "jpn_Jpan"

    inputs = tokenizer(article, return_tensors="pt", padding=True).to(device)
    max_length = 1000

    if target_lang == 'vi':
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("vie_Latn")
    elif target_lang == 'ja':
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("jpn_Jpan")
    else:
        return jsonify({'error': 'Unsupported target language'}), 400

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    for placeholder, english_term in replaced_terms.items():
        translated_text = translated_text.replace(placeholder, english_term)

    end_time = time.time()
    inference_time = end_time - start_time
    write_to_google_sheet(sheet_name, data.get('text'), translated_text, source_lang, target_lang, current_device, inference_time)

    print(f"Inference Time: {inference_time:.2f} seconds")
    return jsonify({'translated_text': translated_text})


@app.route('/log_to_gg_sheet', methods=['POST'])
def log_to_sheet():
    # Nhận dữ liệu từ POST request
    data = request.get_json()
    text = data.get('text')  # Chỉ lấy văn bản dịch
    row_number = data.get('row_number')  # Lấy số dòng

    if not text:
        return jsonify({'error': 'No translated text provided'}), 400

    client = authenticate_google_sheets()
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

    sheet.update_cell(row_number, 8, text)  # Cột H

    return jsonify({'message': 'Log written to Google Sheets successfully!', 'row_number': row_number}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
