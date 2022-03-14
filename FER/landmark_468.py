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
names=["f04","m04"]
#names=["f01","f02","f03","m01","m02","m03"]
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


'''def tokutei(landmarks,img):
    listx =[]
    listy = []
    for y in tokuteiy:
        listy.append(landmarks[y][1])
    for x in tokuteix:
        listx.append(landmarks[x][0])
    lm =(((np.sum(img[listy[0]-2:listy[1]+2,listx[0]:listx[1]],axis=2)/3))/255.0)
    le = (((np.sum(img[listy[2]:listy[3]+2,listx[2]-1:listx[3]+1],axis=2)/3))/255.0)
    rm=(((np.sum(img[listy[4]-2:listy[5]+2,listx[4]:listx[5]],axis=2)/3))/255.0)
    re =(((np.sum(img[listy[6]:listy[7]+2,listx[6]-1:listx[7]+1],axis=2)/3))/255.0)
    mo = (((np.sum(img[listy[8]-4:listy[9]+4,listx[8]-1:listx[9]+1],axis=2)/3))/255.0)
    no = (((np.sum(img[listy[10]:listy[11]+3,listx[10]-25:listx[11]+25],axis=2)/3))/255.0)
    ue =[len(le),len(re),len(lm),len(rm)]
    sita = [len(mo),len(no)]
    a = int(round(np.max(ue)*2))+6
    b = int(round(np.max(sita)*2.5))+6
    new_array = np.zeros((a,224))
    new_array_b = np.zeros((b,224))
    new_array[3:3+lm.shape[0],80:80+lm.shape[1]] = lm
    new_array[lm.shape[0]+5:lm.shape[0]+5+le.shape[0],80:80+le.shape[1]] = le
    new_array[3:3+rm.shape[0],le.shape[1]+110:le.shape[1]+110+rm.shape[1]] =  rm
    new_array[rm.shape[0]+5:rm.shape[0]+5+re.shape[0],le.shape[1]+110:le.shape[1]+110+re.shape[1]] =  re
    new_array_b[:no.shape[0],90:90+no.shape[1]] = no
    new_array_b[no.shape[0]+3:no.shape[0]+3+mo.shape[0],90:90+mo.shape[1]] = mo
    #allfe = np.concatenate([le,re,mo])

    new_array = new_array.reshape((a,224,1))
    new_array_b = new_array_b.reshape((b,224,1))
    return a,b,new_array,new_array_b'''
    
actor  =["01","02","03","04","05","06","07","08","09","10","11","12","13","17","19","20","22","23"]
actor = ["20210616","20210617","20210618","20210619"]
emotion_expr = ["neural","hukai","kai"]
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
            #x,y,z = euler(file)
            s = kaiten(0,0,0)
            img = face_recognition.load_image_file(file)
            if not os.path.exists(f"D/landmark/{name}/"):
                os.makedirs(f"D/landmark/{name}/")
            face_location = face_recognition.face_locations(img,0,"cnn")
            #img = cv2.resize(img,(224,224))
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
            #mp_drawing.draw_landmarks(board, face_landmarks,mp_face_mesh.FACEMESH_CONTOURS,None,mp_drawing_styles
            #.get_default_face_mesh_tesselation_style())
            for p in range(len(landmarks)):
                board[int((1-w)*112+w*(landmarks[p][1])):int((1-w)*112+w*(landmarks[p][1]))+1,int((1-q)*112+(landmarks[p][0])*q):int((1-q)*112+(landmarks[p][0])*q)+1,:] =[1]
            

            
            imageio.imwrite("D/landmark/{}/{}.jpg".format(name,count),board)
        #plt.imshow(img)
        #fig.savefig(f'ha_j/{name}/{label}/lan/{count}landmark.jpg')
            count +=1
