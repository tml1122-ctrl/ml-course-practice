#@title load minst data set
import numpy as np
import pandas as pd
import pylab as pl
from tensorflow.keras.utils import to_categorical

np.random.seed(10)

## ## Method 2: load minst data set

from keras.datasets import mnist
(train_image,train_label),(test_image, test_label) = mnist.load_data()

print(train_image.shape) #60000, 28, 28

print(train_label.shape) #60000, 1

Train_image = train_image.reshape(60000, 784).astype('float32')
Test_image = test_image.reshape(10000, 784).astype('float32')
print('Train_image:',Train_image.shape)
print('Test_image:',Test_image.shape)
Train_image_normalize = Train_image/255  #0--255
Test_image_normalize = Test_image/255

train_label_onehot = to_categorical(train_label)
test_label_onehot = to_categorical(test_label)
train_label_onehot[:5]


# === 繪圖函數 ===
import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf() #design image size
    fig.set_size_inches(2,2) #design image size
    plt.imshow(image, cmap = 'binary')
    plt.show()

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig =plt.gcf()
    fig.set_size_inches(10,10)
    if num>25: num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1) #Generate 5*5 subgraph
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction)>0:
            title+=",predict=" + str(prediction[idx])
            ax.set_title(title, fontsize=10) #Setting subgraph title and size
            ax.set_xticks([]) #don't show the ticks
            ax.set_yticks([]) #don't show the ticks
            idx+=1
            plt.show()

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','validation'], loc = 'upper left') #legend(說明)
    plt.show()

#@title === 建立第一個模型 ===
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential() #generate a linear stack model
model.add(Dense(units = 256, input_dim = 784, kernel_initializer='normal', activation='relu'))
#units : number of Hidden-Layer Neurons(神經元)
#input_dim : number of Input-Layer Neurons
#kernel_initializer = 'normal' : initialize weight and bias by Normal Distribution(常態分布)
#activation : activation

# Generate Ouput-Layer
model.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'softmax'))
print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# loss : lost function
# optimizer : optimizer function
# metrics : evaluative metrics(指標)

train_history = model.fit(
    x = Train_image_normalize,
    y = train_label_onehot,
    validation_split = 0.2,
    epochs = 10,
    batch_size = 200,
    verbose = 2)
# x : features
# y : real label
# validation_split : split 'train' and 'test'
# epochs : number of cycle
# batch_size = number of batch per epochs
# verbose : display form,
# 0 = none, 1 = All
# 2 = Epoch and Metrics(指標), 3 = Epoch


# === 顯示訓練過程與測試結果 ===
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss','val_loss')
scores = model.evaluate(Test_image_normalize, test_label_onehot)
print()
print('accuracy = ', scores[1])

"""這是一個兩層神經網路：  
輸入層：784 個輸入（28×28）  
隱藏層：256 個神經元 + ReLU 激勵函數  
輸出層：10 個神經元（代表數字 0～9）+ softmax 輸出機率  

訓練方式:    
拿 80% 資料訓練，20% 資料驗證 (validation)。  
訓練 10 次迭代（epoch）。  
每次批次處理 200 筆資料。  
"""

#@title === 預測與誤差分析 ===
prediction = np.argmax(model.predict(Test_image), axis=-1)
prediction
plot_images_labels_prediction(test_image, test_label, prediction, idx = 340)
pd.crosstab(test_label, prediction, rownames = ['label'], colnames = ['predict'])
df = pd.DataFrame({'label':test_label, 'predict':prediction})
df[(df.label==5) & (df.predict==6)]
plot_images_labels_prediction(test_image, test_label, prediction, 1378, 1)

"""分析模型錯誤的部分。  
np.argmax(...) 把 one-hot 機率轉回「預測數字」。  
pd.crosstab(...) 做混淆矩陣（看哪些數字容易搞混）。  
df[(df.label==5)&(df.predict==6)] 找出「真的是5但預測成6」的樣本。  
"""

#@title === 重新訓練 + 測試第二個模型 ===
model = Sequential()
model.add(Dense(units = 1000, input_dim = 784, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy',
optimizer = 'adam', metrics = ['accuracy'])
# loss : lost function
# optimizer : optimizer function
# metrics : evaluative metrics(指標)

train_history = model.fit(x = Train_image_normalize, y = train_label_onehot, validation_split = 0.2,
epochs = 10, batch_size = 200, verbose = 2)
show_train_history(train_history,'accuracy','val_accuracy')
scores = model.evaluate(Test_image_normalize, test_label_onehot)
print()
print('accuracy = ',scores[1])

"""隱藏層神經元從 256 → 1000  
新增 Dropout(0.5)：訓練時隨機丟掉一半神經元，防止過擬合。

結果:  
accuracy = 0.9783  (第一個模型)  
accuracy = 0.9805  (第二個模型)  
→ 代表 增加神經元數量 + Dropout 防止過擬合，  
可以讓模型在測試集表現更好。
"""