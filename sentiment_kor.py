from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import json

# 데이터 불러오기
train_data = pd.read_csv("nsmc/ratings_train.txt", sep="\t").dropna()
test_data = pd.read_csv("nsmc/ratings_test.txt", sep="\t").dropna()

x_train = train_data['document']
y_train = train_data['label'].values
x_test = test_data['document']
y_test = test_data['label'].values

# 토크나이징
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)

x_train_pad = pad_sequences(x_train_seq, maxlen=100)
x_test_pad = pad_sequences(x_test_seq, maxlen=100)

# 모델 구성 (여기만 수정한 게 핵심!)
model = Sequential([
    Input(shape=(100,)),
    Embedding(input_dim=10000, output_dim=32),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_pad, y_train, epochs=3, batch_size=64, validation_split=0.2)

# 저장
# model.save("sentiment_model.keras")
model.save("sentiment_model.keras", save_format="keras_v3")  # keras v3 명시 (Keras 3.x 이상 전용)

# tokenizer 저장
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)