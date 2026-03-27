#@title run

import os
import pathlib
import random
import string
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.translate.meteor_score import meteor_score

# 設定 Keras 後端
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization

# 下載 METEOR 所需資源
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- 1. 資料處理 ---
text_file = "/content/English2TraChinese.txt"

with open(text_file, encoding='utf-8') as f:
    lines = f.read().split("\n")[:-1]

text_pairs = []
for line in lines:
    parts = line.split("\t")
    if len(parts) != 2: continue
    eng, chi = parts
    # 中文字元化：將 "你好" 變為 "你 好 "
    chi_chars = ' '.join(list(chi))
    chi_seq = "[start] " + chi_chars + " [end]"
    text_pairs.append((eng, chi_seq))

# 隨機打散並分割資料集
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

# --- 2. 向量化設定 ---
vocab_size = 15000
sequence_length = 50 # 針對短句調整
batch_size = 64

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "").replace("]", "")

def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
)
chi_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

train_eng_texts = [pair[0] for pair in train_pairs]
train_chi_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
chi_vectorization.adapt(train_chi_texts)

def format_dataset(eng, chi):
    eng = eng_vectorization(eng)
    chi = chi_vectorization(chi)
    return ({"encoder_inputs": eng, "decoder_inputs": chi[:, :-1]}, chi[:, 1:])

def make_dataset(pairs):
    eng_texts, chi_texts = zip(*pairs)
    dataset = tf_data.Dataset.from_tensor_slices((list(eng_texts), list(chi_texts)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

# --- 3. 模型定義 (Transformer) ---


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)

    def compute_mask(self, inputs, mask=None):
        return ops.not_equal(inputs, 0)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm_1, self.layernorm_2 = layers.LayerNormalization(), layers.LayerNormalization()

    def call(self, inputs, mask=None):
        padding_mask = ops.cast(mask[:, None, :], dtype="int32") if mask is not None else None
        attn_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attn_output)
        return self.layernorm_2(proj_input + self.dense_proj(proj_input))

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm_1, self.layernorm_2, self.layernorm_3 = layers.LayerNormalization(), layers.LayerNormalization(), layers.LayerNormalization()

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        padding_mask = ops.minimum(ops.cast(mask[:, None, :], dtype="int32"), causal_mask) if mask is not None else None
        out_1 = self.layernorm_1(inputs + self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask))
        out_2 = self.layernorm_2(out_1 + self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask))
        return self.layernorm_3(out_2 + self.dense_proj(out_2))

    def get_causal_attention_mask(self, inputs):
        input_shape = ops.shape(inputs)
        i = ops.arange(input_shape[1])[:, None]
        j = ops.arange(input_shape[1])
        mask = ops.cast(i >= j, dtype="int32")
        return ops.tile(ops.reshape(mask, (1, input_shape[1], input_shape[1])), [input_shape[0], 1, 1])

# 組合模型
embed_dim, latent_dim, num_heads = 256, 2048, 8
encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

transformer = keras.Model([encoder_inputs, decoder_inputs], decoder([decoder_inputs, encoder_outputs]))

# --- 4. 訓練 ---
transformer.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# 作業建議：為了看見效果，epochs 至少設為 30。
transformer.fit(train_ds, epochs=30, validation_data=val_ds)

# --- 5. 評估與 METEOR 計算 ---
chi_vocab = chi_vectorization.get_vocabulary()
chi_index_lookup = dict(zip(range(len(chi_vocab)), chi_vocab))

def decode_sequence(input_sentence):
    tokenized_input = eng_vectorization([input_sentence])
    states_value = encoder(tokenized_input)
    decoded_sentence = "[start]"
    for i in range(20):
        tokenized_target = chi_vectorization([decoded_sentence])[:, :-1]
        predictions = decoder([tokenized_target, states_value])
        sampled_token_index = ops.convert_to_numpy(ops.argmax(predictions[0, i, :])).item(0)
        sampled_token = chi_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]": break
    return decoded_sentence

print("\n--- 隨機 30 句翻譯評估 ---")
total_meteor = 0
for i in range(30):
    pair = random.choice(test_pairs)
    eng_text = pair[0]
    # 原始參考答案（移除標籤並轉回連續字串）
    ref_raw = pair[1].replace("[start]", "").replace("[end]", "").replace(" ", "")

    # 模型預測
    pred_raw = decode_sequence(eng_text).replace("[start]", "").replace("[end]", "").replace(" ", "")

    # METEOR 分數 (參考答案 vs 預測答案)
    # 將字串轉為字元列表
    score = round(meteor_score([list(ref_raw)], list(pred_raw)), 4)
    total_meteor += score

    print(f"{i+1}. 原文: {eng_text}")
    print(f"   參考: {ref_raw}")
    print(f"   預測: {pred_raw} (METEOR: {score})")

print(f"\n平均 METEOR 分數: {total_meteor / 30:.4f}")