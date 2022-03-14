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
lists = ["hap_0","hap_left45","hap_right45"]
names = ["f01","f02","f03","f04","m01","m02","m03","m04"]
lists = ["hap_0"]
left_eye = np.array([36, 37, 38, 39, 40, 41])
right_eye = np.array([42, 43, 44, 45, 46, 47])
mouth = np.array([48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])
f = np.concatenate([left_eye,right_eye,mouth])
for name in names:
    for l in lists:
        path = f"D:/BaiduNetdiskDownload/py/OpenFace_2.2.0_win_x64/kyou/FEDB/{name}/{name}_{l}.mp4"

        '''cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_width= cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite("1.mp4", frame)
            else:
                break'''
        board = np.zeros((224,224,3))
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        count = 0
        out = cv2.VideoWriter(f'{name}{l}.mp4', fourcc, fps, (width, height))
        title_font = ImageFont.truetype("UDDigiKyokashoN-B.ttc",72)
        font = ImageFont.truetype("UDDigiKyokashoN-B.ttc", 48)
        class_name = ["笑顔","真顔"]
        while True:
            try:
                if not cap.isOpened():
                    break
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    l  = cv2.resize(frame_rgb,(224,224))
                    face_location_ori = face_recognition.face_locations(frame_rgb,0,"cnn")
                    landmarks = extract_face_landmarks(l)
                    landmarks_ori = extract_face_landmarks(frame_rgb)

                    for face in face_location_ori:
                        t,r,b,le = face
                        face_ori = frame[t:b,le:r]
                        #pil_image = Image.fromarray(face)
                    #for p in landmarks:
                     #   l[p[1]-2:p[1]+2,p[0]-2:p[0]+2,:] =[255,0,0]
                    for p in landmarks_ori:
                        frame_rgb[p[1]-2:p[1]+2,p[0]-2:p[0]+2,:] =[255,0,0]
                    for p in range(len(landmarks)):
                        if p not in f:
                            continue
                            #l[landmarks[p][1]-3:landmarks[p][1]+3,landmarks[p][0]-3:landmarks[p][0]+3,:] =[255,0,0]
                        else:
                            board[landmarks[p][1]:landmarks[p][1]+2,landmarks[p][0]:landmarks[p][0]+2,:] =[1,1,1]                    
                    face_location = face_recognition.face_locations(l,0,"cnn")
                    for face in face_location:
                        top,right,bottom,left = face
                        face = l[top:bottom,left:right]
                        #pil_image = Image.fromarray(face)
                    face =tf.image.resize(board,(112,112))
                    face = np.reshape(face,(1,112,112,3))     
                    cnn_input = face
                    cv2.rectangle(frame_rgb,(le,t),(r,b+10),(0,255,0),2)
                    result = model(cnn_input)
                    pil_image = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_image)
                    if np.argmax(result,axis=1)[0] == 0:
                        draw.text((le,t-90),"判定結果:",font=title_font,fill=(255,0,0))
                        draw.text((le+312,t-90),str(class_name[np.argmax(result,axis=1)[0]]),font=title_font,fill=(224,81,81))
   
                    else:
                        draw.text((le,t-90),"判定結果:",font=title_font,fill=(255,0,0))
                        draw.text((le+312,t-90),str(class_name[np.argmax(result,axis=1)[0]]),font=title_font,fill=(11,179,30))
                        #draw.text((le,t-90),f"判定結果:"+str(class_name[np.argmax(result,axis=1)[0]]),font=title_font,fill=(250,110,100))
                    draw.text((le,b+10),f"笑顔:"+str(round(result[0][0].numpy()*100,3))+"%",font =font,fill=(224,81,81))
                    draw.text((le,b+60),f"真顔:"+str(round(result[0][1].numpy()*100,3))+"%",font =font,fill=(11,179,30))


                    rgb_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    out.write(rgb_image)
                else:
                    break
            except:
                continue
        
        cap.release()
        out.release()