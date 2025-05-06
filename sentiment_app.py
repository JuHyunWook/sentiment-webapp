import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# 모델 로딩 (.keras 포맷)
# model = tf.keras.models.load_model("sentiment_model.keras")
# model = tf.keras.models.load_model("sentiment_model_v2.keras")
# model = tf.keras.models.load_model("sentiment_model_fixed.keras")
model = tf.keras.models.load_model("sentiment_model.h5")

# 토크나이저 로딩 (JSON 포맷)
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# 예측 함수
def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100)
    score = float(model.predict(padded)[0][0])  # float으로 변환
    return score

# Streamlit 페이지 설정
st.set_page_config(page_title="한글 감정 분석기", layout="centered")

st.markdown("## 🎭 감정 분석 웹앱")
st.markdown("### 🎬 한글 영화 리뷰를 입력하면 AI가 감정을 예측합니다.")
st.markdown("---")

# 입력창
text_input = st.text_area("✍️ 리뷰를 입력해 주세요", placeholder="예: 이 영화 너무 감동적이었어요!")

if st.button("🔍 감정 분석하기"):
    if text_input.strip() == "":
        st.warning('리뷰를 입력해 주세요.')
    else:
        score = predict_sentiment(text_input)

        st.markdown("#### 📊 예측 결과")
        st.progress(min(max(score, 0.0), 1.0))  # 예외 방지용 범위 고정

        if score >= 0.5:
            st.success(f"👍 긍정 리뷰입니다! (확률: {score:.2f})")
        else:
            st.error(f"👎 부정 리뷰입니다. (확률: {score:.2f})")