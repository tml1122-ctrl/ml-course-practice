import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# ========== 1. 資料集載入與前處理 ==========
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((50000, 32 * 32 * 3))
x_test = x_test.reshape((10000, 32 * 32 * 3))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ========== 2. 建立模型 ==========
def build_model(use_bn=False):
    model = Sequential()
    model.add(Input(shape=(32*32*3,)))
    model.add(Dense(256))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(10, activation='softmax'))
    return model

# ========== 3. 優化器設定 (每次建立新實例) ==========
optimizers = {
    "SGD": lambda: SGD(learning_rate=0.01),
    "Momentum": lambda: SGD(learning_rate=0.01, momentum=0.9),
    "RMSProp": lambda: RMSprop(learning_rate=0.001),
    "Adam": lambda: Adam(learning_rate=0.001)
}

# ========== 4. 訓練與記錄結果 ==========
results = {}
EPOCHS = 10  # 先縮短實驗時間
BATCH_SIZE = 128

for name, opt_fn in optimizers.items():
    for use_bn in [False, True]:
        tag = f"{name}_{'BN' if use_bn else 'NoBN'}"
        print(f"\n===== 訓練模型: {tag} =====")

        model = build_model(use_bn)
        opt = opt_fn()  # 每次都建立新的 optimizer
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2
        )

        results[tag] = history

# ========== 5. 畫圖 ==========
for metric in ["accuracy", "loss"]:
    plt.figure(figsize=(10, 6))
    for name in optimizers.keys():
        for use_bn in [False, True]:
            tag = f"{name}_{'BN' if use_bn else 'NoBN'}"
            plt.plot(results[tag].history["val_" + metric], label=f"{tag}")
    plt.title(f"Validation {metric.capitalize()} Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

print("✅ 實驗完成！BN vs NoBN & 各優化器比較圖已繪出。")