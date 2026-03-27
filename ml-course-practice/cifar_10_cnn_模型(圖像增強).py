from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# -----------------------------------------------------
# 1. 載入與預處理資料
# -----------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 將像素值壓縮到 [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot 編碼標籤
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# -----------------------------------------------------
# 2. 建立模型 (含 BN + Dropout)
# -----------------------------------------------------
def create_model():
    model = Sequential([
        # 卷積層 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 卷積層 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 全連接層
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # 輸出層
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------------------------------
# 3. 定義資料增強 (Data Augmentation)
# -----------------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,        # 隨機旋轉角度
    width_shift_range=0.1,    # 水平平移
    height_shift_range=0.1,   # 垂直平移
    horizontal_flip=True,     # 水平翻轉
)
datagen.fit(train_images)

# -----------------------------------------------------
# 4. EarlyStopping 防止過訓
# -----------------------------------------------------
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -----------------------------------------------------
# 5. 訓練有增強的模型
# -----------------------------------------------------
model_with_augmentation = create_model()
history_with_augmentation = model_with_augmentation.fit(
    datagen.flow(train_images, train_labels, batch_size=128),
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[es],
    verbose=1
)

# -----------------------------------------------------
# 6. 訓練無增強的模型
# -----------------------------------------------------
model_without_augmentation = create_model()
history_without_augmentation = model_without_augmentation.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[es],
    verbose=1
)

# -----------------------------------------------------
# 7. 比較 Loss 和 Accuracy
# -----------------------------------------------------
def plot_comparison(history1, history2, label1, label2):
    plt.figure(figsize=(12,5))

    # Loss Comparison
    plt.subplot(1,2,1)
    plt.plot(history1.history['loss'], label=f'{label1} 訓練損失')
    plt.plot(history1.history['val_loss'], label=f'{label1} 驗證損失')
    plt.plot(history2.history['loss'], label=f'{label2} 訓練損失')
    plt.plot(history2.history['val_loss'], label=f'{label2} 驗證損失')
    plt.title('訓練與驗證損失比較')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Comparison
    plt.subplot(1,2,2)
    plt.plot(history1.history['accuracy'], label=f'{label1} 訓練準確率')
    plt.plot(history1.history['val_accuracy'], label=f'{label1} 驗證準確率')
    plt.plot(history2.history['accuracy'], label=f'{label2} 訓練準確率')
    plt.plot(history2.history['val_accuracy'], label=f'{label2} 驗證準確率')
    plt.title('訓練與驗證準確率比較')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_comparison(history_with_augmentation, history_without_augmentation, '有增強', '無增強')

# -----------------------------------------------------
# 8. 儲存模型
# -----------------------------------------------------
model_with_augmentation.save('cifar10_cnn_with_augmentation.h5')
model_without_augmentation.save('cifar10_cnn_without_augmentation.h5')
print("✅ 模型已儲存為 cifar10_cnn_with_augmentation.h5 和 cifar10_cnn_without_augmentation.h5")

'''
# -----------------------------------------------------
# 9. 預測並顯示示範
# -----------------------------------------------------
def predict_and_display(model, image, label):
    plt.imshow(image)
    plt.axis('off')
    pred = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(pred)
    print("🔮 預測類別:", label[predicted_label])

# 載入有增強的模型
model = load_model('cifar10_cnn_with_augmentation.h5')
print("\n有增強模型預測：")
predict_and_display(model, test_images[0], 0)

# 載入無增強的模型
model = load_model('cifar10_cnn_without_augmentation.h5')
print("\n無增強模型預測：")
predict_and_display(model, test_images[0], 1)
'''

# -----------------------------------------------------
# 9. 預測並顯示示範
# -----------------------------------------------------
def predict_and_display(model, image, label):
    plt.imshow(image)
    plt.axis('off')
    pred = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(pred)
    print("🔮 預測類別:", label[predicted_label])

class_names = ['飛機', '汽車', '鳥類', '貓', '鹿', '狗', '青蛙', '馬', '船', '卡車']

# 隨機索引 0~9999圖片索引
idx = np.random.randint(0, len(test_images))
img = test_images[idx]

# 載入有增強的模型
model = load_model('cifar10_cnn_with_augmentation.h5')
print("\n有增強模型預測：")
predict_and_display(model, img, class_names)

# 載入無增強的模型
model = load_model('cifar10_cnn_without_augmentation.h5')
print("\n無增強模型預測：")
predict_and_display(model, img, class_names)