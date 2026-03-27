import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

# === 1. 定義九種顏色與數據下載 ===
COLOR_DICT = {
    'Black':  [0, 0, 0],
    'White':  [255, 255, 255],
    'Red':    [255, 0, 0],
    'Orange': [255, 165, 0],
    'Yellow': [255, 255, 0],
    'Green':  [0, 255, 0],
    'Blue':   [0, 0, 255],
    'Indigo': [75, 0, 130],
    'Purple': [128, 0, 128]
}
COLOR_NAMES = list(COLOR_DICT.keys())
COLOR_VALUES = list(COLOR_DICT.values())

print("正在下載並準備 MNIST 數據...")
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

# === 2. 強化版數據生成：隨機背景 + 九色文字 ===
def make_advanced_color_mnist(images, labels):
    colored_images = []
    digit_color_labels = []

    print(f"正在處理 {len(images)} 筆彩色化數據...")
    for img in images:
        # 隨機從 9 色中挑 2 個不重複的索引 (一個給文字，一個給背景)
        c_font_idx, c_bg_idx = np.random.choice(len(COLOR_NAMES), 2, replace=False)

        font_rgb = COLOR_VALUES[c_font_idx]
        bg_rgb = COLOR_VALUES[c_bg_idx]

        # 建立底色背景 (28x28x3)
        c_img = np.full((28, 28, 3), bg_rgb, dtype=np.uint8)

        # 找出數字部分的 Mask (原始值 > 128)
        mask = img > 128

        # 將數字部分染上字體顏色
        c_img[mask] = font_rgb

        colored_images.append(c_img)
        digit_color_labels.append(c_font_idx) # 目標是辨識「字體顏色」

    return np.array(colored_images) / 255.0, np.array(labels), np.array(digit_color_labels)

# 執行生成
train_images, train_labels_digit, train_labels_color = make_advanced_color_mnist(x_train_raw, y_train_raw)
test_images, test_labels_digit, test_labels_color = make_advanced_color_mnist(x_test_raw, y_test_raw)

# Label 轉 One-hot (數字10類，顏色9類)
train_digit_oh = to_categorical(train_labels_digit, 10)
train_color_oh = to_categorical(train_labels_color, 9)
test_digit_oh = to_categorical(test_labels_digit, 10)
test_color_oh = to_categorical(test_labels_color, 9)

# === 3. 建立多任務 CNN 模型 ===
input_img = Input(shape=(28, 28, 3), name='main_input')

# 特徵提取層 (CNN)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)

# 雙輸出頭
out_digit = Dense(10, activation='softmax', name='digit_output')(x)
out_color = Dense(9, activation='softmax', name='color_output')(x) # 輸出為 9

model = Model(inputs=input_img, outputs=[out_digit, out_color])

# 編譯模型
model.compile(
    optimizer='adam',
    loss={
        'digit_output': 'categorical_crossentropy',
        'color_output': 'categorical_crossentropy'
    },
    metrics={
        'digit_output': 'accuracy',
        'color_output': 'accuracy'
    }
)

# === 4. 開始訓練 ===
print("開始訓練強化版模型...")
history = model.fit(
    train_images,
    {'digit_output': train_digit_oh, 'color_output': train_color_oh},
    validation_split=0.2,
    epochs=10, # 隨機背景難度較高，跑 10 次
    batch_size=128
)

# === 訓練結果視覺化 ===
def show_multitask_history(history):
    plt.figure(figsize=(12, 5))

    # 1. 繪製準確率 (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['digit_output_accuracy'], label='Digit Train Acc')
    plt.plot(history.history['val_digit_output_accuracy'], label='Digit Val Acc')
    plt.plot(history.history['color_output_accuracy'], label='Color Train Acc')
    plt.plot(history.history['val_color_output_accuracy'], label='Color Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 2. 繪製損失函數 (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Total Train Loss')
    plt.plot(history.history['val_loss'], label='Total Val Loss')
    plt.title('Model Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 執行繪圖
show_multitask_history(history)

# === 5. 驗證與視覺化預測結果 ===
def quick_predict(index):
    img = test_images[index:index+1]
    predictions = model.predict(img, verbose=0)

    pred_digit = np.argmax(predictions[0])
    pred_color = np.argmax(predictions[1])

    plt.figure(figsize=(2, 2))
    plt.imshow(test_images[index])
    plt.title(f"Real: {test_labels_digit[index]}, {COLOR_NAMES[test_labels_color[index]]}\n"
              f"Pred: {pred_digit}, {COLOR_NAMES[pred_color]}", fontsize=9)
    plt.axis('off')
    plt.show()

# 隨機挑選 5 張測試結果
print("\n展示測試集預測結果：")
for _ in range(5):
    quick_predict(np.random.randint(0, 10000))