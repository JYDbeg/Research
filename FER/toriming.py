from PIL import Image,ImageDraw
import face_recognition
import glob
import os
names=["f04","m04"]
from face_recognition.api import face_landmarks
names=["f01","f02","f03","m01","m02","m03","f04","m04"]
#l = ["happiness","neural"]
labels = ["happiness","happinessleft45","happinessright45"]
expr = ["angcl","angop","discl","disop","exc","fea","hap","rel","sad","sle","sur"]
count = 0

'''for name in names:
    for label in labels:
        for file in glob.glob("ha_j/{}/{}/no90kao/*.jpg".format(name,label)):
            image=face_recognition.load_image_file(file)
            face_location = face_recognition.face_locations(image,0,"cnn")
            for face in face_location:
                top,right,bottom,left = face
                face = image[top:bottom,left:right]
                pil_image = Image.fromarray(face)
                if not os.path.exists("ha_j/{}/{}/onlyface".format(name,label)):
                    os.makedirs("ha_j/{}/{}/onlyface".format(name,label))
                pil_image.save(f"ha_j/{name}/{label}/onlyface/{count}.jpg")
                count+=1'''
'''for name in names:
    for label in labels:
        for file in glob.glob("ha_v/{1}/{1}/{1}/*.jpg".format(name,label)):
            image=face_recognition.load_image_file(file)
            face_landmarks_list = face_recognition.face_landmarks(image)
            pil_image = Image.fromarray(image)
            d = ImageDraw.Draw(pil_image)
            for face in face_landmarks_list:
                for fea  in face.keys():
                    d.line(face[fea],width=5)

                pil_image.save(f"ha_j/{name}/{label}/onlyface/{count}landmark.jpg")
                count+=1'''
'''for name in names:
    for label in labels:
        for file in glob.glob("ha_v/{}/{}/*.jpg".format(name,label)):
            image=face_recognition.load_image_file(file)
            face_location = face_recognition.face_locations(image,0,"cnn")
            for face in face_location:
                top,right,bottom,left = face
                face = image[top:bottom,left:right]
                pil_image = Image.fromarray(face)
                if not os.path.exists("ha_v/{}/{}/onlyface".format(name,label)):
                    os.makedirs("ha_v/{}/{}/onlyface".format(name,label))
                pil_image.save(f"ha_v/{name}/{label}/onlyface/{count}.jpg")
                count+=1'''
names=["normal","smile"]
names = ["tanaka","iwata","miymoto","nisio"]
expr=["negative","positive","normal"]
dir_path = "newdata/postive-"
for name in names:
    for ex in expr:
        for file in glob.glob(dir_path+name+'/'+ex+'/*.jpg'):
            image=face_recognition.load_image_file(file)
            face_location = face_recognition.face_locations(image,0,"cnn")
            for face in face_location:
                top,right,bottom,left = face
                face = image[top:bottom,left:right]
                pil_image = Image.fromarray(face)
                if not os.path.exists(f"newdata/faceonly/{name}/{ex}"):
                    os.makedirs(f"newdata/faceonly/{name}/{ex}/")
                pil_image.save(f"newdata/faceonly/{name}/{ex}/{count}.jpg")
                count+=1
                
