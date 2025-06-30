import streamlit as st
from audio_recorder_streamlit import audio_recorder
from audio_utils import predict_emotion
import tempfile

st.title("Speech Emotion Recognition (SER)")

st.write("🎤 Rekam suara atau unggah file audio (.wav) untuk prediksi emosi suara.")

# Rekam audio dari mic
audio_bytes = audio_recorder(text="Klik untuk merekam suara", pause_threshold=2.0)

if audio_bytes:
    # Simpan audio hasil rekaman ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    st.audio(tmp_path, format="audio/wav")

    if st.button("Prediksi Emosi dari Mic"):
        with st.spinner("Memproses..."):
            label, confidence = predict_emotion(tmp_path)

        if label:
            st.success(f"Emosi terdeteksi: **{label}**")
        else:
            st.error("Gagal memproses audio dari mic.")

# Upload audio manual
audio_file = st.file_uploader("Atau unggah file audio .wav", type=["wav"])

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())
    
    st.audio("temp_audio.wav", format="audio/wav")
    
    if st.button("Prediksi Emosi dari File"):
        with st.spinner("Memproses..."):
            label, confidence = predict_emotion("temp_audio.wav")
        
        if label:
            st.success(f"Emosi terdeteksi: **{label}**")
        else:
            st.error("Gagal memproses file audio.")
