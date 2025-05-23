import streamlit as st
import requests
import json
# Thiết lập cấu hình trang để làm cho giao diện rộng hơn
st.set_page_config(page_title="Dịch Văn Bản với NLLB-200", layout="wide")

# Thay thế URL cho Flask API và Log API theo môi trường của bạn
# FLASK_API_URL = "http://103.67.198.56:5000/translate"
DIFY_API_URL = "https://api.dify.ai/v1/workflows/run"
API_KEY = "app-Bd7vCBGdUFAIOG0XA4D4bi42"
LOG_API_URL = "http://103.67.198.56:5000/log_to_gg_sheet"

# Kiểm tra và tạo các biến trong session_state nếu chưa có
if 'source_lang' not in st.session_state:
    st.session_state.source_lang = "vi"
if 'target_lang' not in st.session_state:
    st.session_state.target_lang = "ja"
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'row_number' not in st.session_state:
    st.session_state.row_number = None

st.title("Dịch Văn Bản với NLLB-200")
st.write("Nhập văn bản và chọn ngôn ngữ đích để dịch:")

languages = {
    "ja": "Nhật",
    "vi": "Việt"
}

source_lang = st.selectbox("Chọn ngôn ngữ nguồn", ["vi", "ja"],
                           index=["vi", "ja"].index(st.session_state.source_lang),
                           format_func=lambda x: languages[x])
st.session_state.source_lang = source_lang

if source_lang == "vi":
    st.session_state.target_lang = "ja"
else:
    st.session_state.target_lang = "vi"

# Nhập văn bản nguồn
source_text = st.text_area("Văn bản nguồn", "")

# Kiểm tra xem API có được gọi chưa
if 'api_called' not in st.session_state:
    st.session_state.api_called = False

# API Dịch
if st.button("Dịch"):
    if source_text:
        payload = {
            "inputs": {
                "text": source_text,
                "source_lang": st.session_state.source_lang,
                "target_lang": st.session_state.target_lang,
                "sheet_name": "LOG_COMMON"
            },
            "response_mode": "blocking",
            "user": "abc-123"
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }

        # Gửi yêu cầu API
        response = requests.post(DIFY_API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            response_data = response.json()

            # Lấy dữ liệu từ "outputs" trong response
            outputs = response_data.get('data', {}).get('outputs', {})
            if outputs:
                # Phân tích chuỗi JSON trong trường 'response' để lấy các giá trị cần thiết
                try:
                    response_json = json.loads(outputs.get('response', '{}'))
                    translated_text = response_json.get('translated_text')
                    row_number = response_json.get('row_number')

                    # Hiển thị kết quả dịch
                    if translated_text:
                        st.session_state.translated_text = translated_text
                        st.session_state.row_number = row_number
                        st.session_state.api_called = True

                        # Lưu vào lịch sử dịch
                        st.session_state.history.append({
                            'source_text': source_text,
                            'translated_text': translated_text,
                            'row_number': row_number,
                            'type': 'dich'  # Thêm thông tin loại (dịch)
                        })

                        # Giới hạn số lượng lịch sử dịch, chỉ giữ lại 30 mục gần nhất
                        if len(st.session_state.history) > 30:
                            st.session_state.history = st.session_state.history[-30:]

                        # Hiển thị kết quả dịch trên UI
                        st.write(f"Dịch: {translated_text}")
                    else:
                        st.write("Không có văn bản dịch.")
                except json.JSONDecodeError:
                    st.write("Lỗi phân tích dữ liệu từ API.")
            else:
                st.write("Không có dữ liệu đầu ra từ API.")
        else:
            st.write("Lỗi khi gọi API:", response.text)
    else:
        st.write("Vui lòng nhập văn bản để dịch.")
# Hiển thị lịch sử dịch trong Sidebar
with st.sidebar:
    st.header("Lịch sử Dịch")
    if len(st.session_state.history) > 0:
        for entry in st.session_state.history:
            # Thêm màu nền khác nhau cho Dịch và Chỉnh sửa
            if entry['type'] == 'dich':
                background_color = "#D3F9D8"  # Màu nền cho Dịch (nhạt)
            else:
                background_color = "#F7E1D7"  # Màu nền cho Chỉnh sửa (nhạt cam)

            st.markdown(f"""
                <div style="background-color:{background_color}; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                    <p style="font-size: 12px;"><strong>Nguồn:</strong> {entry['source_text']}</p>
                    <p style="font-size: 12px;"><strong>Đích:</strong> {entry['translated_text']}</p>
                    <p style="font-size: 12px;"><strong>Số dòng trong Sheets:</strong> {entry['row_number']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.write("Chưa có lịch sử dịch nào.")

# Nút để ghi nội dung chỉnh sửa vào Google Sheets
if st.session_state.api_called:
    show_extra_input = st.checkbox("Thêm nội dung chỉnh sửa")

    if st.session_state.translated_text:
        st.write("Văn bản đã dịch:")
        st.write(st.session_state.translated_text)

    if show_extra_input:
        extra_input = st.text_area("Nội dung muốn chỉnh sửa", "")

        if st.button("Cập nhật chỉnh sửa vào Google Sheets"):
            if st.session_state.translated_text and st.session_state.row_number:
                if extra_input:
                    log_payload = {
                        'text': extra_input,
                        'row_number': st.session_state.row_number  # Gửi row_number vào API
                    }

                    # Gửi yêu cầu POST tới API log_to_gg_sheet để ghi log vào Google Sheets
                    log_response = requests.post(LOG_API_URL, json=log_payload)

                    if log_response.status_code == 200:
                        st.write("Nội dung chỉnh sửa đã được ghi vào Google Sheets.")

                        # Sau khi thành công, thêm vào lịch sử
                        st.session_state.history.append({
                            'source_text': source_text,
                            'translated_text': extra_input,  # Thêm bản dịch hoặc nội dung chỉnh sửa
                            'row_number': st.session_state.row_number,
                            'type': 'chinh_sua'  # Thêm thông tin loại (chỉnh sửa)
                        })

                        # Giới hạn số lượng lịch sử dịch, chỉ giữ lại 30 mục gần nhất
                        if len(st.session_state.history) > 30:
                            st.session_state.history = st.session_state.history[-30:]

                        # Làm mới giao diện để cập nhật thanh lịch sử
                        st.rerun()

                    else:
                        st.write("Lỗi khi ghi vào Google Sheets:", log_response.text)
                else:
                    st.write("Vui lòng nhập nội dung chỉnh sửa.")
            else:
                st.write("Vui lòng dịch văn bản trước khi ghi vào Google Sheets.")
