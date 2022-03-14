import numpy as np
import cv2
from mlxtend.image import extract_face_landmarks
import glob
import face_recognition
import imageio
import matplotlib.pyplot as plt
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
names = ["normal","smile"]
for name in names:
    print(name)
    for index,item in enumerate(glob.glob(f"test/*.jpg")):

        img = cv2.imread(item,0)
        cl1 = clahe.apply(img)
        cv2.imwrite(f"test//{index}.jpg",cl1)
