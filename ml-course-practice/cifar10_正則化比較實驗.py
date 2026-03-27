

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print(f"✅ TensorFlow version: {tf.__version__}")
tf.config.list_physical_devices('GPU')

# ==========================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((50000, 32*32*3))
x_test = x_test.reshape((10000, 32*32*3))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ==========================================
def create_model(reg_type=None, reg_value=1e-4, dropout_rate=None):
    inputs = Input(shape=(32*32*3,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)

    # 加入正則化
    if reg_type == "L1":
        x = layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l1(reg_value))(x)
    elif reg_type == "L2":
        x = layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(reg_value))(x)
    elif reg_type == "L1L2":
        x = layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l1_l2(l1=reg_value, l2=reg_value))(x)
    else:
        x = layers.Dense(128, activation='relu')(x)

    # Dropout
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
configs = [
    ("No Regularization", None, None),
    ("L1 Regularization", "L1", None),
    ("L2 Regularization", "L2", None),
    ("Dropout (0.5)", None, 0.5)
]

histories = {}
results = []

EPOCHS = 20  # 減少訓練時間，可改成 50

for name, reg, drop in configs:
    print(f"\n🚀 Training model: {name}")
    model = create_model(reg_type=reg, dropout_rate=drop)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS, batch_size=128, verbose=2
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    results.append([name, score[0], score[1]])
    histories[name] = history

# ==========================================
df = pd.DataFrame(results, columns=["Model", "Val Loss", "Val Accuracy"])
print("\n📊 結果比較：")
print(df)

# ==========================================
plt.figure(figsize=(12,5))
for name, _, _ in configs:
    plt.plot(histories[name].history['val_loss'], label=f"{name}")
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
for name, _, _ in configs:
    plt.plot(histories[name].history['val_accuracy'], label=f"{name}")
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ==========================================
print("""
📘 小結論：
- 無正則化：通常訓練 loss 很低，但驗證 loss 很高（過擬合）
- L1/L2：權重被限制，模型比較穩定但收斂較慢
- Dropout：最常見的正則化手段，能顯著降低過擬合
""")