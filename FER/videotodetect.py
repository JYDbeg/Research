from re import L
import cv2
import imageio
import config
import matplotlib.pyplot as plt
from mlxtend.image import extract_face_landmarks
import glob
import tensorflow as tf
from train import get_model
model = get_model()
model.load_weights(filepath = config.save_model_dir)
import face_recognition
from PIL import ImageFont, ImageDraw, Image
import numpy as np
expr = ["angcl","angop","discl","disop","exc","fea","hap","rel","sad","sle","sur"]
class_name = ["怒（閉口）","怒（開口）","嫌悪（閉口","嫌悪（開口）","興奮","恐れ","笑顔","真顔","リラックス","悲しみ","眠気","驚き"]
names = ["f01","f02","f03","f04","m01","m02","m03","m04"]
left_eye = np.array([36, 37, 38, 39, 40, 41])
right_eye = np.array([42, 43, 44, 45, 46, 47])
mouth = np.array([48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])
rinkaku = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
f = np.concatenate([left_eye,right_eye,mouth])
tokuteiy = [19,17,37,40,24,22,43,46,51,57,28,33]
tokuteix =[17,21,36,39,22,26,42,45,48,54,31,35]
def tokutei(landmarks,img):
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
    new_array = new_array.reshape((a,224,1))
    new_array_b = new_array_b.reshape((b,224,1))
    return a,b,new_array,new_array_b
for name in names:
    for l in expr:
        model.clear_memory()
        path = f"somedirectory/{name}/{name}_{l}_0.mp4"
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        count = 0
        out = cv2.VideoWriter(f'{name}{l}.mp4', fourcc, fps, (width, height))
        title_font = ImageFont.truetype("UDDigiKyokashoN-B.ttc",28)
        font = ImageFont.truetype("UDDigiKyokashoN-B.ttc", 24)
        class_name = ["ネガティブ","ポジティブ","真顔"]
        while True:
            if not cap.isOpened():
                break
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                s  = cv2.resize(frame_rgb,(224,224))
                face_location_ori = face_recognition.face_locations(frame_rgb,0,"cnn")
                landmarks = extract_face_landmarks(s)
                a,bs,new_array,new_array_b =tokutei(landmarks,s)
                landmarks_ori = extract_face_landmarks(frame_rgb)
                board = np.zeros((224,224,1))
                diff = landmarks[33]-112
                for face in face_location_ori:
                    t,r,b,le = face
                    face_ori = frame[t:b,le:r]
                for p in landmarks_ori:
                    frame_rgb[p[1]-2:p[1]+2,p[0]-2:p[0]+2,:] =[255,0,0]
                for p in range(len(landmarks)):
                    if p in rinkaku:
                        continue
                    else:
                        board[landmarks[p][1]-diff[1]-15:landmarks[p][1]-diff[1]-14,landmarks[p][0]-diff[0]:landmarks[p][0]-diff[0]+1,:] =[1]      
                board[:a,:] = new_array
                board[-bs:,:] = new_array_b
    
                face_location = face_recognition.face_locations(s,0,"cnn")
                for face in face_location:
                    top,right,bottom,left = face
                    face = s[top:bottom,left:right]
                   
                face =tf.image.resize(board,(112,112))
                face = np.reshape(face,(1,112,112,1))     
                cnn_input = face
                cv2.rectangle(frame_rgb,(le,t),(r,b+10),(0,0,255),2)
                result= model(cnn_input,ifTest=True)
                pil_image = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(pil_image)
                if np.argmax(result,axis=1)[0] == 0:
                    draw.text((le,b+10),"判定結果:",font=title_font,fill=(255,0,0))
                    draw.text((le+312,b+10),str(class_name[np.argmax(result,axis=1)[0]]),font=title_font,fill=(224,81,81))

                else:
                    draw.text((le,b+10),"判定結果:",font=title_font,fill=(255,0,0))
                    draw.text((le+312,b+10),str(class_name[np.argmax(result,axis=1)[0]]),font=title_font,fill=(11,179,30))
                for i in range(len(class_name)):
                    draw.text((0,20*i),class_name[i]+str(round(result[0][i]*100,3))+"%",font =font,fill=(224,81,81))
                rgb_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                out.write(rgb_image)
            else:
                break
        cap.release()
        out.release()
