from PIL.Image import new
import imageio
from mlxtend.image import extract_face_landmarks
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import glob
import numpy as np
from tensorflow.python.keras.preprocessing.image import array_to_img
import os
import cv2

names=["f01","f02","f03","f04","m01","m02","m03","m04"]
ls = ["happiness","neural"]
labels = ["happiness","happinessleft45","happinessright45"]
left_eye = np.array([36, 37, 38, 39, 40, 41])
right_eye = np.array([42, 43, 44, 45, 46, 47])
mouth = np.array([48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])
rinkaku = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
f = np.concatenate([left_eye,right_eye,mouth])
count = 0
expr = ["angcl","angop","discl","disop","exc","fea","hap","rel","sad","sle","sur"]
tokuteiy = [19,17,37,40,24,22,43,46,51,57,28,33]
tokuteix =[17,21,36,39,22,26,42,45,48,54,31,35]
left_mayu =[17,18,19,20,21]
right_mayu =[22,23,24,25,26]
nose = [27,28,29,30]
nose_bottom =[31,32,33,34,35]
mouse = [48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]
def tokutei(landmarks,img):
    listx =[]
    listy = []
    img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.Canny(img,70,100)
    

    for y in tokuteiy:
        listy.append(landmarks[y][1])
    for x in tokuteix:
        listx.append(landmarks[x][0])
    lm =(((img[listy[0]-1:listy[3]+3,listx[0]:listx[1]]))/255.0)
    lm = cv2.resize(lm,(40,24))
    rm=((img[listy[0]-1:listy[3]+3,listx[4]:listx[5]])/255.0)
    rm = cv2.resize(rm,(40,24))
    mo = ((img[listy[8]-4:listy[9]+4,listx[8]-1:listx[9]+1])/255.0)
    mo = cv2.resize(mo,(48,20))
    a = 45#int(round(np.max(ue)*2))+6
    b = 55#int(round(np.max(sita)*2.5))+6
    new_array = np.zeros((a,224))
    new_array_b = np.zeros((b,224))
    new_array[3:3+lm.shape[0],80:80+lm.shape[1]] = lm
    new_array[3:3+rm.shape[0],130:130+rm.shape[1]] =  rm
    new_array_b[3:mo.shape[0]+3,100:100+mo.shape[1]] = mo
    new_array = new_array.reshape((a,224,1))
    new_array_b = new_array_b.reshape((b,224,1))
    return a,b,new_array,new_array_b
asdfg = np.arange(17,68,1)
from PIL import Image    
import face_recognition 

for index,file in enumerate(glob.glob("D/20210616/Column/*.jpg")):
    img = face_recognition.load_image_file(file)
    face_location = face_recognition.face_locations(img,0,"cnn")
    for face in face_location:
        top,right,bottom,left = face
    y = bottom - top
    x = right - left
    multiply = img.shape[0]/img.shape[1]
    print(multiply)
    landmarks = extract_face_landmarks(img)
    sad = np.mean(landmarks[asdfg], axis=0)
    a,b,new_array,new_array_b =tokutei(landmarks,img)
    board = np.zeros((224,224,1))
    q = 112*multiply/x
    w =112/y
    diffx = int(sad[0])-112
    diffy = int(sad[1])-112
    left_center = np.mean(landmarks[left_eye], axis=0)
    right_center = np.mean(landmarks[right_eye], axis=0)
    left_center_mayu = np.mean(landmarks[left_mayu], axis=0)
    right_center_mayu = np.mean(landmarks[right_mayu], axis=0)
    nose_center = np.mean(landmarks[nose], axis=0)
    nose__center = np.mean(landmarks[nose_bottom], axis=0)
    mouse_center = np.mean(landmarks[mouse], axis=0)
    for p in range(len(landmarks)):
        if p  in rinkaku:
            continue
        landmarks[p][1] = landmarks[p][1]-diffy
        landmarks[p][0] = landmarks[p][0]-diffx
    for p in range(len(landmarks)):
        if p  in rinkaku:
            continue
        else:
            board[int((1-w)*112+w*(landmarks[p][1])):int((1-w)*112+w*(landmarks[p][1]))+2,int((1-q)*112+(landmarks[p][0])*q):int((1-q)*112+(landmarks[p][0])*q)+2,:] =[1]
    
    board[:a,:] = new_array
    board[-b:,:] = new_array_b
    
    imageio.imwrite("test/1/{}.jpg".format(count),board)
    count +=1
