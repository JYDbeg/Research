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
import mediapipe as mp
from augle import euler
from dist import kaiten
'''
names=["f01","f02","f03","f04","m01","m02","m03","m04"]
expr = ["angcl","angop","discl","disop","exc","fea","hap","rel","sad","sle","sur"]   
actor  =["01","02","03","04","05","06","07","08","09","10","11","12","13","17","19","20","22","23"]
actor = ["20210616","20210617","20210618","20210619"]
emotion_expr = ["neural","hukai","kai"]
'''
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
names=["normal","smile"]
import face_recognition
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh: 
    for name in names:
        for index,file in enumerate(glob.glob("0829/{}/*.jpg".format(name))):
            s = kaiten(0,0,0)
            img = face_recognition.load_image_file(file)
            if not os.path.exists(f"D/landmark/{name}/"):
                os.makedirs(f"D/landmark/{name}/")
            face_location = face_recognition.face_locations(img,0,"cnn")
            if len(face_location)>0:
                for face in face_location:
                    top,right,bottom,left = face
                    face = img[top-50:bottom+50,left-50:right+50]
                y = bottom - top+100
                x = right - left+100
            results = face_mesh.process(face)
            board = np.zeros((224,224,1))
            for face_landmarks in results.multi_face_landmarks:
                landmarks =[]
            ih,iw,ic = [y,x,3]
            for idx,lm in enumerate(face_landmarks.landmark):
                landmarks.append([abs(int(lm.x*iw*s[0][0])+int(lm.y*ih*s[0][1])+int(lm.z*iw*s[0][2])), abs(int(lm.x*iw*s[1][0])+int(lm.y*ih*s[1][1])+int(lm.z*iw*s[1][2]))])
            y_m = np.max(landmarks[:][1])
            x_m = np.max(landmarks[:][0])
            q = 112/x_m
            w =112/y_m
            diffx = landmarks[1][0]-112
            diffy = landmarks[1][1]-112
            for p in range(len(landmarks)):
                landmarks[p][1] = landmarks[p][1]-diffy
                landmarks[p][0] = landmarks[p][0]-diffx
            for p in range(len(landmarks)):
                board[int((1-w)*112+w*(landmarks[p][1])):int((1-w)*112+w*(landmarks[p][1]))+1,int((1-q)*112+(landmarks[p][0])*q):int((1-q)*112+(landmarks[p][0])*q)+1,:] =[1]
            imageio.imwrite("D/landmark/{}/{}.jpg".format(name,count),board)
            count +=1
