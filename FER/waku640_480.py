def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj 
import torch.jit   
torch.jit.script_method = script_method 
torch.jit.script = script
import cv2
import numpy as np
import face_recognition
import torch
import torch.nn as nn

#from PIL import ImageFont, ImageDraw, Image
import numpy as np
from poolformer import poolformer_s12
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from signal_handler import Handler
import numpy as np
import fft_filter
import hr_calculator
import mediapipe as mp
#from mttcan import run_on_video

mp_face_detection = mp.solutions.face_detection
face_detection =mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.3)
freqs_min = 0.8
freqs_max = 1.8
ecg_mask = np.zeros([100,100,3],dtype=np.uint8)
#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
#mp_face_mesh = mp.solutions.face_mesh
#drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
def get_hr(ROI, fps):
    signal_handler = Handler(ROI)
    blue, green, red = signal_handler.get_channel_signal()
    matrix = np.array([blue, green, red])
    component = signal_handler.ICA(matrix, 3)
    fft, freqs = fft_filter.fft_filter(component[0], freqs_min, freqs_max, fps)
    heartrate_1 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    fft, freqs = fft_filter.fft_filter(component[1], freqs_min, freqs_max, fps)
    heartrate_2 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    fft, freqs = fft_filter.fft_filter(component[2], freqs_min, freqs_max, fps)
    heartrate_3 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    return (abs(heartrate_1) + abs(heartrate_2) + abs(heartrate_3)) / 3

model = poolformer_s12()
model.head = nn.Sequential(nn.Linear(512,250),nn.ReLU(),nn.Linear(250,2))

#model.head = nn.Sequential(nn.Linear(512,250),nn.ReLU(),nn.Linear(250,12))
model.load_state_dict(torch.load("myFirstModel.pth"))

#use_camera = int(input("choose your camera ID:"))
cap_file = cv2.VideoCapture("01-02-06-01-01-01-04.mp4")
fps = cap_file.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v') 
#out = cv2.VideoWriter('edited_video.mp4', fourcc, fps,(640,480)) 


wakuface = cv2.imread("waku1.png",-1)
wakuface =cv2.resize(wakuface,(640,480))

pointbig = cv2.imread("pointbig.png",-1)
ecg = cv2.imread("heart.png",-1)

pointbig = cv2.resize(pointbig,(20,20))
hyoujo = ["smile","angry","sleepy","normal","sad"]
b_list = [290,230,190,140]
a_list = [500,570,630]
last = [381,105]
co = 577-361
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
emotion_stress = 0
max_emotion_stress = 1000
pos = [218,557,428,768]
ROI = []
heartrate = 0
bpm = 0
first =1
count_f = 0
fade = 1
alpha = 1
target_layers = [model.norm]
cam = XGradCAM(model = model,target_layers = target_layers)
#detector = dlib.get_frontal_face_detector()
frames = []
heartrates = []
for i in range(301):
    heartrates.append((i,250))
heartrates =np.array(heartrates)
'''face_mesh =mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)'''
while cap_file.isOpened():
    ret, frame =cap_file.read()
    if not ret:
        continue

    frame = cv2.resize(frame,(640,480))
    facetodetect = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(facetodetect)
    facetodetectcopy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #facetodetectgcam = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_location = face_recognition.face_locations(facetodetect,0,"cnn")
    
    if not len(face_location):
        continue
    for face in face_location:
        top,right,bottom,left = face
        face = facetodetect[top:bottom+20,left:right]
    #dects = detector(frame)
    #if not len(dects):
    #    continue
    #for face in dects:
        #left = face.left()
        #right = face.right()
        #top = face.top()
        #bottom = face.bottom()
        h = bottom - top
        w = right - left
        
        roi = frame[top + h // 10 * 5:top + h // 10 * 7, left + w // 9 * 3:left + w // 9 * 5]
        ROI.append(roi)
        print(len(ROI))
        if len(ROI) == 300:
            heartrate = get_hr(ROI, fps)
            heartrates[-1] =np.array((300,250-int(heartrate)))
            ROI.pop(0)        
    '''if len(frames) == 300:
        target_to_estimate_heartbeat = np.array(frames)
        timess,BPM = run_on_video(fps = fps, frames = target_to_estimate_heartbeat)
        bpm = np.mean(BPM)
        frames = frames[30:]'''
    hf = 243-h
    wf = 235-w
    hhf = int(0.5*hf)
    hwf = int(0.5*wf)
    #face = facetodetect[top:bottom,left:right]
    #facey = bottom-top
    #facex = right-left
    f = facetodetect[max(0,top-hhf):bottom+hhf,max(0,left-hwf):right+hwf]
    '''results = face_mesh.process(facetodetectcopy)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
            image=facetodetectcopy,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())'''
    lm = facetodetectcopy[top:bottom,left:right]
    lm = cv2.resize(lm,(150,150))
    face = cv2.resize(face,(224,224))
    
    gradcam_face = face[:]/255.0
    face_ = face.transpose(2,0,1)
    face_ =face_/255.0
    face_ = np.expand_dims(face_,0)
    face_ = torch.FloatTensor(face_)
    with torch.no_grad():
        data = face_.to(device)
        outputs = model(data).to('cpu').detach().numpy().copy()
    #print(outputs)
    '''if "heartrate"  in locals():
        if heartrate>80:
            outputs[6:] = outputs[6:]*100
        else:
            outputs[:6] = outputs[:6]*100'''
    #outputs = np.argmax(outputs)
    valence = outputs[0][0]
    arousal = outputs[0][1]
    target= [ClassifierOutputTarget(0)]
    grayscale_cam= cam(input_tensor = data,targets = target)
    grayscale_cam = grayscale_cam[0, :]
    visualization1 = show_cam_on_image(gradcam_face, grayscale_cam, use_rgb=True)
    target= [ClassifierOutputTarget(1)]
    grayscale_cam= cam(input_tensor = data,targets = target)
    grayscale_cam = grayscale_cam[0, :]
    visualization2 = show_cam_on_image(gradcam_face, grayscale_cam, use_rgb=True)
    visualization =visualization1*0.5+visualization2*0.5
    visualization = np.array(visualization,dtype=np.uint8)
    visualization = cv2.cvtColor(visualization,cv2.COLOR_RGB2BGR)
    visualization =cv2.resize(visualization,(150,150))
    
    #a = outputs%3
    #b = outputs//3
    x_ = 40+valence*130#b_list[b]/2+np.random.randint(-5,5)
    y_ = 470-arousal*180#a_list[a]/1.5+np.random.randint(-5,5)
    if x_>105 and x_<115:
        wariai = x_/145
        emotion_stress -=0.1*wariai*abs(x_-last[1])
        emotion_stress = max(0,emotion_stress)
    elif x_>=115:
        wariai = x_/145
        emotion_stress -=0.4*wariai*abs(x_-last[1])
        emotion_stress = max(0,emotion_stress)
    elif x_<105 and x_>95:
        wariai  =70/x_
        emotion_stress += 0.1*wariai*abs(last[1]-x_)
        emotion_stress = min(max_emotion_stress,emotion_stress)
    else:
        wariai  =70/x_
        emotion_stress += 0.3*wariai*abs(last[1]-x_)
        emotion_stress = min(max_emotion_stress,emotion_stress)
    wariai = emotion_stress/max_emotion_stress
    x = wariai*co
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #frame[:720,:1280] = frame[:720,:1280] *(1-wakuoo[:,:,3:]/255)+wakuoo[:,:,:3]*(wakuoo[:,:,3:]/255)
    #frame[:720,:1280] = frame[:720,:1280] *(1-wakuver2[:,:,3:]/255)+wakuver2[:,:,:3]*(wakuver2[:,:,3:]/255)
    #face = cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
    f = cv2.cvtColor(f,cv2.COLOR_RGB2BGR)
    lm = cv2.cvtColor(lm,cv2.COLOR_RGB2BGR)
    frame[:480,:640] = [0,0,0]
    frame[:480,:640] = frame[:480,:640] *(1-wakuface[:,:,3:]/255)+wakuface[:,:,:3]*(wakuface[:,:,3:]/255)
    f_y,f_x,c = f.shape
    l_y,l_x,c = lm.shape
    g_y,g_x,c = visualization.shape
    frame[3:3+g_y,100:100+g_x] = visualization
    frame[:100,:100] = frame[:100,:100] * (1-ecg[:,:,3:]/255)+ecg[:,:,:3]*(ecg[:,:,3:]/255)
    
    if fade :
        diff_alpha =heartrate/900
        alpha -= diff_alpha
        if alpha<=0:
            fade = 0
            alpha = 0
        frame[:100,:100] = cv2.addWeighted(frame[:100,:100],alpha,ecg_mask,1-alpha,0)
    else:
        diff_alpha = heartrate/900
        alpha+=diff_alpha
        if alpha>=1:
            fade = 1
            alpha = 1
        frame[:100,:100] = cv2.addWeighted(frame[:100,:100],alpha,ecg_mask,1-alpha,0)
    #frame[:100,int(count_f*5/3):int(count_f*5/3)+2] = [0,0,0]
    frame[3:3+f_y,400:400+f_x] = f
    frame[3:3+l_y,250:250+l_x] = lm
    frame[386:422,361+int(x):577] = [0,0,0]
    if wariai>0.5 and wariai<0.7:
        frame[386:422,361:361+int(x)] = [0,255,255]
    elif  wariai>0.7 :
        frame[386:422,361:361+int(x)] = [0,0,255]
    a = round(valence,2)
    b = round(arousal,2)
    cv2.putText(frame,f'valence={str(a)},arousal={str(b)} ',org=(200,280),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_4)
    cv2.putText(frame,f'postive : valence<2',org=(200,310),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_4)
    cv2.putText(frame,f'negative : valence>=2',org=(200,340),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_4)

                        #draw.text((le,t-90),f"判定結果:"+str(class_name[np.argmax(result,axis=1)[0]]),font=title_font,fill=(250,110,100))
    #frame[:720,:1280] = frame[:720,:1280] *(1-bar[:,:,3:]/255)+bar[:,:,:3]*(bar[:,:,3:]/255)
    #frame[:720,:1280] = frame[:720,:1280] *(1-barb[:,:,3:]/255)+barb[:,:,:3]*(barb[:,:,3:]/255)
    #frame[50:150,50:150]=frame[50:150,50:150]*(1-waku[:,:,3:]/255)+waku[:,:,:3]*(waku[:,:,3:]/255)
    #index = np.random.randint(0,9)
    #px = zahyoux[index]
    #index = np.random.randint(0,9)
    #py = zahyouy[index]
    diffx = x_-last[1]
    diffy = y_-last[0]
    multix = diffx/5
    multiy = diffy/5
    cv2.rectangle(frame, (400 + w // 9 * 3,4 + h // 10 * 5), (400 + w // 9 * 5, 4 + h // 10 * 7),
                        color=(0, 255, 255))
    if last[0]<380 and last[1] >110:
        hyougazo = cv2.imread(hyoujo[0]+".png",-1)
        frame[280:380,100:200] = frame[280:380,100:200] *(1-hyougazo[:,:,3:]/255)+hyougazo[:,:,:3]*(hyougazo[:,:,3:]/255)
    elif last[0]>380 and last[1] >110:
        hyougazo = cv2.imread(hyoujo[2]+".png",-1)
        frame[320:420,100:200] = frame[320:420,100:200] = frame[320:420,100:200] *(1-hyougazo[:,:,3:]/255)+hyougazo[:,:,:3]*(hyougazo[:,:,3:]/255)
    elif 420>last[0] and 360<last[0] and 80<last[1] and last[1]<130:
        hyougazo = cv2.imread(hyoujo[3]+".png",-1)
        frame[280:380,50:150] = frame[280:380,50:150] = frame[280:380,50:150] *(1-hyougazo[:,:,3:]/255)+hyougazo[:,:,:3]*(hyougazo[:,:,3:]/255)
    elif last[1]<100 and last[0]<380:
        hyougazo = cv2.imread(hyoujo[1]+".png",-1)
        frame[280:380,30:130] = frame[280:380,30:130] = frame[280:380,30:130] *(1-hyougazo[:,:,3:]/255)+hyougazo[:,:,:3]*(hyougazo[:,:,3:]/255)
    elif last[1]<100 and last[0]>380:
        hyougazo = cv2.imread(hyoujo[4]+".png",-1)
        frame[360:460,20:120] = frame[360:460,20:120] = frame[360:460,20:120] *(1-hyougazo[:,:,3:]/255)+hyougazo[:,:,:3]*(hyougazo[:,:,3:]/255)
    last= [int(last[0]+multiy),int(last[1]+multix)]
    #cv2.putText(frame, '{:.1f}bpm'.format(bpm), (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, '{:.1f}bpm'.format(heartrate), (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, '{:.1f}fps'.format(fps), (200, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    frame[last[0]:last[0]+20,last[1]:last[1]+20]=frame[last[0]:last[0]+20,last[1]:last[1]+20]*(1-pointbig[:,:,3:]/255)+pointbig[:,:,:3]*(pointbig[:,:,3:]/255)
    cv2.rectangle(frame,(40,290),(170,470),(88,77,155),2)
    #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    #frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #pil_image = Image.fromarray(frame_rgb)
    #draw = ImageDraw.Draw(pil_image)
    #draw.text((100,100),f'valence={b},arousal={a}',font=title_font,fill=(255,0,0))
    #draw.text((100,200),f'positive : valence<2 negative : valence>=2',font=title_font,fill=(255,0,0))
    #draw.text((100,300),f'適当に作った',font=title_font,fill=(255,0,0))
    #rgb_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if len(ROI) == 299:
        heartrates[:,0]-=1
        heartrates[:-1] = heartrates[1:]
        #print(heartrates)
        cv2.polylines(frame,[heartrates], False, (255,0, 0))
    #out.write(frame)
    count_f += heartrate/50
    count_f= count_f%60
    cv2.imshow("v",frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap_file.release()
#out.release()
cv2.destroyAllWindows()
