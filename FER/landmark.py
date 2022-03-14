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
    lm =(((np.sum(img[listy[0]-2:listy[1]+2,listx[0]:listx[1]],axis=2)/3))<100)
    le = (((np.sum(img[listy[2]:listy[3]+2,listx[2]-1:listx[3]+1],axis=2)/3))<85)
    rm=(((np.sum(img[listy[4]-2:listy[5]+2,listx[4]:listx[5]],axis=2)/3))<100)
    re =(((np.sum(img[listy[6]:listy[7]+2,listx[6]-1:listx[7]+1],axis=2)/3))<85)
    mo = (((np.max(img[listy[8]-4:listy[9]+4,listx[8]-1:listx[9]+1],axis=2)))<125)
    no = (((np.sum(img[listy[10]:listy[11]+3,listx[10]-25:listx[11]+25],axis=2)))<350)
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
    new_array_b[:no.shape[0],80:80+no.shape[1]] = no
    new_array_b[no.shape[0]+3:no.shape[0]+3+mo.shape[0],80:80+mo.shape[1]] = mo
    #allfe = np.concatenate([le,re,mo])

    new_array = new_array.reshape((a,224,1))
    new_array_b = new_array_b.reshape((b,224,1))
    return a,b,new_array,new_array_b'''
lmm = 0.527
rmm = 0.547
mom = 0.482
nom = 0.583    
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
    #lm = lm<0.2
    lm = cv2.resize(lm,(40,24))
    #le = (((np.sum(img[listy[2]:listy[3]+2,listx[2]-1:listx[3]+1],axis=2)/3))/255.0)
    rm=((img[listy[0]-1:listy[3]+3,listx[4]:listx[5]])/255.0)
    #rm = rm<0.2
    rm = cv2.resize(rm,(40,24))
    #re =(((np.sum(img[listy[6]:listy[7]+2,listx[6]-1:listx[7]+1],axis=2)/3))/255.0)
    mo = ((img[listy[8]-4:listy[9]+4,listx[8]-1:listx[9]+1])/255.0)
    #mo = mo<0.5
    mo = cv2.resize(mo,(48,20))
    #no = (((img[listy[10]:listy[11]+3,listx[10]-25:listx[11]+25])/255.0))
    #no = no<0.2
    #no = cv2.resize(no,(72,30))
    #ue =[len(lm),len(rm)]#len(le),len(re),len(lm),len(rm)]
    #sita = [len(mo),len(no)]
    a = 45#int(round(np.max(ue)*2))+6
    b = 55#int(round(np.max(sita)*2.5))+6
    new_array = np.zeros((a,224))
    new_array_b = np.zeros((b,224))
    new_array[3:3+lm.shape[0],80:80+lm.shape[1]] = lm
    #new_array[lm.shape[0]+5:lm.shape[0]+5+le.shape[0],80:80+le.shape[1]] = le
    new_array[3:3+rm.shape[0],130:130+rm.shape[1]] =  rm
    #new_array[rm.shape[0]+5:rm.shape[0]+5+re.shape[0],le.shape[1]+110:le.shape[1]+110+re.shape[1]] =  re
    #new_array_b[:no.shape[0],90:90+no.shape[1]] = no
    new_array_b[3:mo.shape[0]+3,100:100+mo.shape[1]] = mo
    #allfe = np.concatenate([le,re,mo])
    #print(landmarks[17][0],landmarks[26][0])
    new_array = new_array.reshape((a,224,1))
    new_array_b = new_array_b.reshape((b,224,1))
    return a,b,new_array,new_array_b
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
asdfg = np.arange(17,68,1)
x_1 =59
x_2 =161
   
from PIL import Image    
#names=["normal","smile"]
import face_recognition 
#for name in names:
   # for label in expr:

for index,file in enumerate(glob.glob("D/20210616/Column/*.jpg")):

    img = face_recognition.load_image_file(file)
    face_location = face_recognition.face_locations(img,0,"cnn")
    for face in face_location:
        top,right,bottom,left = face
        #face = image[top:bottom,left:right]
    y = bottom - top
    x = right - left
    multiply = img.shape[0]/img.shape[1]
    print(multiply)
    landmarks = extract_face_landmarks(img)
    sad = np.mean(landmarks[asdfg], axis=0)
    a,b,new_array,new_array_b =tokutei(landmarks,img)
    board = np.zeros((224,224,1))
    #s = landmarks[17][0]-landmarks[26][0]
    #prer = ((161-59)+s)//2
    #per = 224/img.shape[0]
    #pery=224/img.shape[1]
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
    '''for p in range(len(landmarks)):
        if p  in rinkaku:
            continue
        if p in left_eye:
            landmarks[p][1] = int((1-w)*left_center[1]+w*(landmarks[p][1]))
            landmarks[p][0] = int((1-q)*left_center[0]+q*(landmarks[p][0]))
        if p in right_eye:
            landmarks[p][1] = int((1-w)*right_center[1]+w*(landmarks[p][1]))
            landmarks[p][0] = int((1-q)*right_center[0]+q*(landmarks[p][0]))
        if p in left_mayu:
            landmarks[p][1] = int((1-w)*left_center_mayu[1]+w*(landmarks[p][1]))
            landmarks[p][0] = int((1-q)*left_center_mayu[0]+q*(landmarks[p][0]))
        if p in right_mayu:
            landmarks[p][1] = int((1-w)*right_center_mayu[1]+w*(landmarks[p][1]))
            landmarks[p][0] = int((1-q)*right_center_mayu[0]+q*(landmarks[p][0]))
        if p in nose:
            landmarks[p][1] = int((1-w)*nose_center[1]+w*(landmarks[p][1]))
            landmarks[p][0] = int((1-q)*nose_center[0]+q*(landmarks[p][0]))
        if p in nose_bottom:
            landmarks[p][1] = int((1-w)*nose__center[1]+w*(landmarks[p][1]))
            landmarks[p][0] = int((1-q)*nose__center[0]+q*(landmarks[p][0]))
        if p in mouse:
            landmarks[p][1] = int((1-w)*mouse_center[1]+w*(landmarks[p][1]))
            landmarks[p][0] = int((1-q)*left_center[0]+q*(landmarks[p][0]))'''
    for p in range(len(landmarks)):
        if p  in rinkaku:
            continue
    
        else:
            board[int((1-w)*112+w*(landmarks[p][1])):int((1-w)*112+w*(landmarks[p][1]))+2,int((1-q)*112+(landmarks[p][0])*q):int((1-q)*112+(landmarks[p][0])*q)+2,:] =[1]
    
    board[:a,:] = new_array
    board[-b:,:] = new_array_b
    
    imageio.imwrite("test/1/{}.jpg".format(count),board)
#plt.imshow(img)
#fig.savefig(f'ha_j/{name}/{label}/lan/{count}landmark.jpg')
    count +=1

'''for index,file in enumerate(glob.glob("ha_alll/{}/{}/neural/augment/*.jpg".format(name,label,label))):
            try:
                if not os.path.exists("augl/{}/neural".format(name)):
                    os.makedirs("augl/{}/neural".format(name))
                img = imageio.imread(file)
                img = cv2.resize(img,(224,224))
                landmarks = extract_face_landmarks(img)
                a,b,new_array,new_array_b =tokutei(landmarks,img)
                board = np.zeros((224,224,1))
                diff = landmarks[33]-122
                for p in range(len(landmarks)):
                    if p  in rinkaku:
                        continue
                
                    else:
                        board[landmarks[p][1]-15-diff[1]:landmarks[p][1]-diff[1]-14,landmarks[p][0]-diff[0]:landmarks[p][0]-diff[0]+1,:] =[1]
                board[:a,:] = new_array
                board[-b:,:] = new_array_b

                imageio.imwrite("augl/{}/neural/{}.jpg".format(name,count),board)
                #plt.imshow(img)
                #fig.savefig(f'ha_j/{name}/{label}/lan/{count}landmark.jpg')
                count +=1
            except:
                continue'''

'''for name in names:
    for label in expr:
        if not os.path.exists("ha_alllll/{}/{}/land/{}".format(name,label,label)):
            os.makedirs("ha_alllll/{}/{}/land/{}".format(name,label,label))
        if not os.path.exists("ha_alllll/{}/{}/land/neural".format(name,label)):
            os.makedirs("ha_alllll/{}/{}/land/neural".format(name,label))'''