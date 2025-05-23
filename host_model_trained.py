from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gspread
from google.oauth2.service_account import Credentials
import datetime
import time
from peft import PeftModel

# Flask app
app = Flask(__name__)

# Google Sheets authentication
CREDENTIALS_FILE = 'test-training-data-245841565d66.json'
SPREADSHEET_ID = '1spg80mi-dbMZX_97DJrv2PQnA_f7pr1Elprvn3CP9WU'
SHEET_NAME = 'LOG_COMMON'
MODEL_NAME = "facebook/nllb-200-1.3B"
model_dir = "./results"

# Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, model_dir)
model.eval()
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# test_sentence = "ボタン"
# baseline_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# baseline_output = baseline_model.generate(test_sentence, max_length=20)
# baseline_translation = tokenizer.batch_decode(baseline_output, skip_special_tokens=True)[0]
#
# # Model sau khi fine-tune
# fine_tuned_output = model.generate(test_sentence, max_length=20)
# fine_tuned_translation = tokenizer.batch_decode(fine_tuned_output, skip_special_tokens=True)[0]
#
# print(f"Baseline Translation: {baseline_translation}")
# print(f"Fine-tuned Translation: {fine_tuned_translation}")


# Authenticate Google Sheets
def authenticate_google_sheets():
    creds = Credentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    return gspread.authorize(creds)


# Load special terms from Google Sheets
def load_special_terms_from_sheet():
    client = authenticate_google_sheets()
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet("JP_TO_ENG")
    data = sheet.get_all_values()
    return {row[0]: row[1] for row in data if len(row) >= 2}  # Cột A -> key, Cột B -> value


# Write log to Google Sheets
def write_to_google_sheet(sheet_name, source_text, target_text, source_lang, target_lang, current_device,
                          inference_time):
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


@app.route('/translate', methods=['POST'])
def translate():
    start_time = time.time()
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    article = data['text']
    source_lang = data.get('source_lang', 'ja')
    target_lang = data.get('target_lang', 'vi')

    current_device = "GPU" if torch.cuda.is_available() else "CPU"
    special_terms = load_special_terms_from_sheet()
    replaced_terms = {}

    # # Replace special terms
    # for term, replacement in special_terms.items():
    #     placeholder = f"[SPECIAL_TERM_{len(replaced_terms)}]"
    #     if term in article:
    #         article = article.replace(term, placeholder)
    #         replaced_terms[placeholder] = replacement

    # Set source and target languages
    lang_map = {
        'ja': 'jpn_Jpan',
        'vi': 'vie_Latn',
        'en': 'eng_Latn'
    }

    if source_lang not in lang_map or target_lang not in lang_map:
        return jsonify({'error': 'Unsupported language pair'}), 400

    tokenizer.src_lang = lang_map[source_lang]
    tokenizer.tgt_lang = lang_map[target_lang]

    # Tokenize input
    inputs = tokenizer(article, return_tensors="pt", padding=True).to(device)
    max_length = 1000

    # Force BOS token
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(lang_map[target_lang])

    # Generate translation
    try:
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        # # Replace placeholders with original terms
        # for placeholder, replacement in replaced_terms.items():
        #     translated_text = translated_text.replace(placeholder, replacement)

        end_time = time.time()
        inference_time = end_time - start_time

        # Log to Google Sheets
        write_to_google_sheet(SHEET_NAME, data['text'], translated_text, source_lang, target_lang, current_device,
                              inference_time)

        return jsonify({'translated_text': translated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
