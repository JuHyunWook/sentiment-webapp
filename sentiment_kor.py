import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
import json

# 데이터 불러오기
train_data = pd.read_csv("nsmc/ratings_train.txt", sep="\t").dropna()
test_data = pd.read_csv("nsmc/ratings_test.txt", sep="\t").dropna()

# 텍스트만 추출
x_train = train_data['document'].values
y_train = train_data['label'].values
x_test = test_data['document'].values
y_test = test_data['label'].values

# 토크나이저
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train)
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)
x_train_pad = pad_sequences(x_train_seq, maxlen=100)
x_test_pad = pad_sequences(x_test_seq, maxlen=100)

# 모델 정의
model = Sequential([
    Input(shape=(100,)),
    Embedding(input_dim=10000, output_dim=32),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_pad, y_train, epochs=3, batch_size=64, validation_split=0.2)
model.evaluate(x_test_pad, y_test)

# 모델 저장 (.keras 형식)
model.save("sentiment_model.keras")

# 토크나이저 저장 (JSON 방식)
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)