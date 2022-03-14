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
import face_recognition 
import cv2
import csv
import mediapipe as mp
num = [58,2,12]
def eular(x,y,n):
    if n ==58:
        a,b = x[1]-y[1],x[2]-y[2]
        c,d = 1,0
        kakudo = np.degrees(np.arccos((a*c+b*d)/(np.sqrt(a**2+b**2)*np.sqrt(d**2+c**2))))
    elif n ==2:
        a,b = x[0]-y[0],x[2]-y[2]
        c,d = 0,1
        kakudo = np.degrees(np.arccos((a*c+b*d)/(np.sqrt(a**2+b**2)*np.sqrt(d**2+c**2))))
    elif n==12:
        a,b = x[0]-y[0],x[1]-y[1]
        c,d = 1,0
        kakudo = np.degrees(np.arccos((a*c+b*d)/(np.sqrt(a**2+b**2)*np.sqrt(d**2+c**2))))
    return kakudo
def kaiten(x, y,z):
    x_a = [[1,0,0],[0,np.cos(np.radians(x)),np.sin(np.radians(x))],[0,-np.sin(np.radians(x)),np.cos(np.radians(x))]]
    y_a = [[np.cos(np.radians(y)),0,-np.sin(np.radians(y))],[0,1,0],[np.sin(np.radians(y)),0,np.cos(np.radians(y))]]
    z_a = [[np.cos(np.radians(z)),np.sin(np.radians(z)),0],[-np.sin(np.radians(z)),np.cos(np.radians(z)),0],[0,0,1]]
    a_a = np.dot(np.dot(x_a,y_a),z_a)
    return a_a
if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    from mpl_toolkits.mplot3d import Axes3D
    # For static images:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(glob.glob("test/sd.jpg")):
            image = cv2.imread(file)
            #image = cv2.resize(image,(40,30))
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.imshow(image)
            plt.show()
        # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                landmarks =[]
                x=[]
                y=[]
                z=[]
                qq = np.zeros((image.shape[0], image.shape[1],3))
                print('face_landmarks:', face_landmarks)
                #mp_drawing.draw_landmarks(image, face_landmarks,mp_face_mesh.FACEMESH_TESSELATION,None,mp_drawing_styles
            #   .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(image, face_landmarks,mp_face_mesh.FACEMESH_CONTOURS,None,mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
                ih,iw,ic = image.shape
                poss=[]
                for idx,lm in enumerate(face_landmarks.landmark):
                    poss.append([lm.x,lm.y,lm.z])
                poss = np.array(poss)
                listdata = []
                poss_mean = np.mean(poss,axis =0)
                for i in poss:
                    listdata.append([np.sqrt((i[0]-poss_mean[0])**2+(i[1]-poss_mean[1])**2+(i[2]-poss_mean[2])**2),np.arccos((i[0]*poss_mean[0]+i[1]*poss_mean[1]+i[2]*poss_mean[2])/(np.sqrt(i[0]**2+i[1]**2+i[2]**2)*np.sqrt(poss_mean[0]**2+poss_mean[1]**2+poss_mean[2]**2)))])
                print(len(listdata))
                #x_ang = eular(poss[150],poss[151],58)    
                #y_ang = eular(poss[33],poss[126],2)
                #z_ang = eular(poss[132],poss[361],12)
                s =kaiten(0,0,0)
                for idx,lm in enumerate(face_landmarks.landmark):
                    '''if idx == 0:
                        x_ = lm.x
                        y_ = lm.y
                        z_ = lm.z
                    with open("test/distx.csv","a",newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([lm.z-z_,lm.y-y_,idx])
                        #f.write("z_dist = "+str(lm.z-z_)+"y_dist ="+str(lm.y-y_)+"idx = "+str(idx)+":"+"\n")
                    with open("test/disty.csv","a",newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([lm.x-x_,lm.z-z_,idx])
                        #f.write("x_dist = "+str(lm.x-x_)+"z_dist ="+str(lm.z-z_)+"idx = "+str(idx)+":"+"\n")
                    with open("test/distz.csv","a",newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([lm.x-x_,lm.y-y_,idx])
                        #f.write("x_dist = "+str(lm.x-x_)+"y_dist ="+str(lm.y-y_)+"idx = "+str(idx)+":"+"\n") '''
                    landmarks.append([int(lm.x*iw*s[0][0])+int(lm.y*ih*s[0][1])+int(lm.z*ih*s[0][2]), abs(int(lm.x*iw*s[1][0])+int(lm.y*ih*s[1][1])+int(lm.z*ih*s[1][2]))])
                    x.append([int(lm.x*iw*s[0][0])+int(lm.y*ih*s[0][1])+int(lm.z*ih*s[0][2])])
                    y.append([int(lm.x*iw*s[1][0])+int(lm.y*ih*s[1][1])+int(lm.z*ih*s[1][2])])
                    z.append([int(lm.x*iw*s[2][0])+int(lm.y*ih*s[2][1])+int(lm.z*ih*s[2][2])])
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(x,y,z,s=10,c="green")
            plt.show()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for p in range(len(landmarks)):
                qq[landmarks[p][1]:landmarks[p][1]+2,landmarks[p][0]:landmarks[p][0]+2,:] =[255,255,255] 
            plt.imshow(qq)
            plt.show()

        rinkaku = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        for file in glob.glob("test/*.jpg"):
            img = face_recognition.load_image_file(file)
            face_location = face_recognition.face_locations(img,0,"cnn")
            #img = cv2.resize(img,(224,224))
            landmarks = extract_face_landmarks(img)
            plt.imshow(img)
            plt.show()
            board = np.zeros((img.shape[1],img.shape[0],1))
            for p in range(len(landmarks)):
                if p  in rinkaku:
                    continue
            
                else:
                    img[landmarks[p][1]:landmarks[p][1]+1,landmarks[p][0]:landmarks[p][0]+1,:] =[255]

                '''with open("test/1.txt","a") as f:
                    f.write(str(landmarks[p])+":"+str(p)+"\n")'''