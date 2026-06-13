import os
import re
import streamlit as st

# =====================================================
# KONFIGURASI
# =====================================================

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 128

LABELS = [
    "APLIKASI_POSITIF",
    "APLIKASI_NEGATIF",
    "INTERFACE_POSITIF",
    "INTERFACE_NEGATIF",
    "LAYANAN_POSITIF",
    "LAYANAN_NEGATIF",
    "KEAMANAN_POSITIF",
    "KEAMANAN_NEGATIF"
]

# Nama file target penyimpanan model di server Streamlit
TARGET_MODEL_NAME = "best_model_indobert.pt"

# ID Google Drive default Anda
GDRIVE_FILE_ID = "1CfMtU7haRgITdEnN3BVsTzB08CbjeDmG"

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# LOAD & DOWNLOAD MODEL
# =====================================================

def _extract_drive_file_id(url_or_id: str) -> str:
    if url_or_id is None:
        return ""
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", url_or_id) and "/" not in url_or_id:
        return url_or_id

    patterns = [
        r"https?://drive\.google\.com/file/d/([A-Za-z0-9_-]+)",
        r"https?://drive\.google\.com/open\?id=([A-Za-z0-9_-]+)",
        r"https?://drive\.google\.com/uc\?id=([A-Za-z0-9_-]+)",
        r"id=([A-Za-z0-9_-]+)"
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    return ""

@st.cache_resource
def load_model_from_drive(drive_link_or_id: str):
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    class IndoBERTClassifier(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.bert = AutoModel.from_pretrained(MODEL_NAME)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            pooled_output = self.dropout(pooled_output)
            logits = self.fc(pooled_output)
            return logits

    file_id = _extract_drive_file_id(drive_link_or_id)
    if not file_id:
        raise RuntimeError("ID file Google Drive tidak valid. Masukkan link atau ID yang benar.")

    if not os.path.exists(TARGET_MODEL_NAME):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            with st.spinner("Mengunduh model dari Google Drive (Proses ini hanya berjalan sekali)..."):
                gdown.download(url, TARGET_MODEL_NAME, quiet=False)
        except Exception as download_error:
            raise RuntimeError(
                f"Gagal mengunduh file dari Google Drive. Pastikan file dapat diakses 'Anyone with the link'. Detail: {download_error}"
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = IndoBERTClassifier(len(LABELS))

    try:
        state_dict = torch.load(TARGET_MODEL_NAME, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as load_error:
        if os.path.exists(TARGET_MODEL_NAME):
            os.remove(TARGET_MODEL_NAME)
        raise RuntimeError(
            f"Gagal memuat bobot model. File mungkin korup atau tidak kompatibel. Detail: {load_error}"
        )

    return tokenizer, model

# =====================================================
# PREDIKSI
# =====================================================

def predict(text, tokenizer, model):
    import torch

    text = preprocess(text)
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )
        probabilities = torch.sigmoid(logits).squeeze()

    return probabilities.cpu().numpy()

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen Multi Label",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("Analisis Sentimen Multi Label IndoBERT")

st.markdown(
    """
    Aplikasi ini menggunakan model IndoBERT
    untuk mendeteksi sentimen berdasarkan aspek:

    - Aplikasi
    - Interface
    - Layanan
    - Keamanan
    """
)

# Input link Google Drive dan tombol untuk memuat model dari Drive
default_drive_link = f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view?usp=sharing"
drive_link = st.text_input("Link Google Drive model", value=default_drive_link)

# Jika session_state kosong tetapi file model sudah ada di server semenjak startup, 
# lakukan auto-load demi kenyamanan pengguna URL Publik.
if "tokenizer" not in st.session_state or "model" not in st.session_state:
    st.session_state.tokenizer = None
    st.session_state.model = None

# Mekanisme Trigger Pemuatan Model
if st.button("Muat Model dari Drive"):
    try:
        tokenizer_loaded, model_loaded = load_model_from_drive(drive_link)
        st.session_state.tokenizer = tokenizer_loaded
        st.session_state.model = model_loaded
        st.success("Model berhasil aktif dan siap digunakan!")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
elif os.path.exists(TARGET_MODEL_NAME) and st.session_state.model is None:
    st.info("Model sudah tersedia di server. Tekan 'Muat Model dari Drive' untuk mengaktifkan tanpa mengunduh ulang.")

# Ambil objek tokenizer & model dari session_state untuk inferensi
tokenizer = st.session_state.tokenizer
model = st.session_state.model

review = st.text_area(
    "Masukkan ulasan pengguna",
    height=200,
    placeholder="Contoh: Aplikasi sangat membantu tetapi tampilannya masih kurang menarik."
)

threshold = st.slider(
    "Threshold Prediksi",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

if st.button("Prediksi"):
    if review.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")
        st.stop()

    if tokenizer is None or model is None:
        st.warning("Model belum aktif. Silakan masukkan link Google Drive yang valid dan ketuk 'Muat Model dari Drive' terlebih dahulu.")
        st.stop()

    probs = predict(review, tokenizer, model)
    st.subheader("Hasil Prediksi")

    for label, prob in zip(LABELS, probs):
        prediction = "✅ Positif/Terdeteksi" if prob > threshold else "❌ Tidak Terdeteksi"
        st.write(f"**{label}** : {prediction}")
        st.progress(float(prob))
        st.caption(f"Probabilitas: {prob:.4f}")
