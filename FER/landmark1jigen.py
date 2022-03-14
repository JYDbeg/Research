import imageio
import matplotlib.pyplot as plt
#from prepare_datafnn import load_and_preprocess_image
import face_recognition
names = ["iwata","miymoto","tanaka","nisio"]
expr=["negative","postive"]
dir_path = "newdata/postive-"
import glob
import numpy as np
import cv2
left_eye =[246,7,161,163,160,144,159,145,158,153,157,154,173,155,55,221,222,65,52,223,224,53,225,46]
right_eye =[398,382,381,384,385,380,386,374,373,387,388,466,390,249,276,445,444,283,282,443,442,295,441,285]
mouth = [61,185,40,39,37,0,267,259,270,409,291,287,57,375,321,405,314,17,84,181,91,146,76,184,74,73,72,11,302,303,304,408,306,307,320,404,315,16,85,180,90,77,62,183,42,41,38,12,268,271,272,407,292,325,319,403,316,15,86,179,89,96,78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
'''names  =["01","02","03","04","05","06","07","08","09","10","11","12","13","17","19","20","22","23"]
expr =["hukai","kai","neural"]
dir_path = "Actor_video/"'''

'''def load_and_preprocess_image(img_path):
    # read pictures
    image = imageio.imread(img_path)
    try:
        results = extract_face_landmarks(image)
        if results ==None:
            face_location = face_recognition.face_locations(image,0,"cnn")
            for face in face_location:
                top,right,bottom,left = face
                image = image[top:bottom,left:right]
            results = extract_face_landmarks(image)
        
        listdata = []
        poss_mean = results[33]
        for i in results:
            listdata.append([np.sqrt((i[0]-poss_mean[0])**2+(i[1]-poss_mean[1])**2),np.arccos((i[0]*poss_mean[0]+i[1]*poss_mean[1])/(np.sqrt(i[0]**2+i[1]**2)
                                                                                                                                        *np.sqrt(poss_mean[0]**2+poss_mean[1]**2)))])
        listdata = np.array(listdata)
        listdata = listdata.flatten()
                
    except:
        return 0'''

import mediapipe as mp
def load_and_preprocess_image(img_path,face_mode = 0):
    # read pictures
    image=face_recognition.load_image_file(img_path)
    face = 0
    if face_mode == 1:
        face_location = face_recognition.face_locations(image,0,"cnn")
        if len(face_location) ==0:
            return 0,0
        for face in face_location:
            top,right,bottom,left = face
            face = image[top:bottom,left:right]
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    #y=bottom-top
    #x = right-left
    #ld= np.zeros((x,y,1))
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(face)
        if not results.multi_face_landmarks:
            return 0,0
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks,mp_face_mesh.FACEMESH_CONTOURS,None,mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
            poss=[]
            for lm in face_landmarks.landmark:
                poss.append([lm.x,lm.y,lm.z])
            
            poss = np.array(poss)
            #for p in range(len(poss)):
            #    ld[poss[p][1]:poss[p][1]+2,poss[p][0]:poss[p][0]+2,:] =[255] 
            listdata = []
            #poss_mean = np.mean(poss,axis =0)
            for i in range(len(left_eye)):
                for d in range(len(left_eye)):
                    if i == d :
                        continue
                    listdata.append(np.linalg.norm(poss[i]-poss[d]))
            for i in range(len(right_eye)):
                for d in range(len(right_eye)):
                    if i ==d :
                        continue
                    listdata.append(np.linalg.norm(poss[i]-poss[d]))
            for i in range(len(mouth)):
                for d in range(len(mouth)):
                    if i == d :
                        continue
                    listdata.append(np.linalg.norm(poss[i]-poss[d]))
            #for i in poss:
                #listdata.append([np.sqrt((i[0]-poss_mean[0])**2+(i[1]-poss_mean[1])**2+
                                            #(i[2]-poss_mean[2])**2),np.arccos((i[0]*poss_mean[0]+i[1]*poss_mean[1]+i[2]*poss_mean[2])/(np.sqrt(i[0]**2+i[1]**2+i[2]**2)
                                                                                                                          #              *np.sqrt(poss_mean[0]**2+poss_mean[1]**2+poss_mean[2]**2)))])
            listdata = np.array(listdata)
            listdata = listdata.flatten()
    return listdata,face
from tqdm import tqdm
count =0
import os
for name in names:
    lis = []
    target =[]
    for ex in expr:
        for file in tqdm(glob.glob(dir_path+name+'/'+ex+'/*.jpg')):
            kekka,face =  load_and_preprocess_image(file,face_mode=1)
            if type(kekka) is int:
                continue
            index = expr.index(ex)
            lis.append(kekka)
            target.append(index)
            #if not os.path.exists(f"newdata/landmark/{name}/{ex}"):
            #    os.makedirs(f"newdata/landmark/{name}/{ex}/")
          #  imageio.imwrite(f"newdata/landmark/{name}/{ex}/{count}.jpg",kekka)
            #count+=1
        
        print(ex,name,len(lis))
    np.savez(dir_path+name+"-distance-nonormal-landmarks",lis,target)

            
'''iwata = np.load('newdata/postive-iwatalandmarks.npz')
miymoto = np.load('newdata/postive-miymotolandmarks.npz')
tanaka = np.load('newdata/postive-tanakalandmarks.npz')
nisio = np.load('newdata/postive-nisiolandmarks.npz')
print(len(nisio["arr_0"]))'''