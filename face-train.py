import os
import cv2
from PIL import Image
import numpy as np
import pickle

faceCascade = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, "images")

img_train =[]
label_train=[]
label_id = {}
count_id = 1
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            pil_image = Image.open(path).convert('L') #gray scale
            pil_image = pil_image.resize((500, 500), Image.ANTIALIAS)
            img_arr = np.array(pil_image, "uint8")
            
            faces = faceCascade.detectMultiScale(img_arr, 1.5, 3)

            person = os.path.basename(root)
            if person not in label_id:
             label_id.setdefault(person, count_id)
             count_id+=1
            for (x, y, w, h) in faces:
                roi = img_arr[y:y+h, x:x+w]
                cv2.imshow("img", roi)
                img_train.append(roi)
                label_train.append(count_id)
                print(f'train {file} success')

with open("labels.pickle", "wb") as f:
    pickle.dump(label_id, f)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(img_train, np.array(label_train))
recognizer.save("train.yml")