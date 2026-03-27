!unzip /content/face_recognition.zip -d /content
!pip install -q mtcnn keras-facenet

import os
os.chdir("/content/face_recognition")
print(os.getcwd())  # 確認是 face_recognition 資料夾

!python face_recognition.py

#@title 顯示圖片
from matplotlib import pyplot as plt
from PIL import Image

test_folder = "/content/face_recognition/test"

for test_file in os.listdir(test_folder):
    test_path = os.path.join(test_folder, test_file)
    img = Image.open(test_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(test_file)
    plt.show()