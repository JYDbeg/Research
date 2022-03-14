import cv2
import glob
import os 
names=["f01","f02","f03","f04","m01","m02","m03","m04"]
labels = ["happiness","neural"]
ang = ["_0"]
expr = ["angcl","angop","discl","disop","exc","fea","hap","rel","sad","sle","sur"]
path = "test"
vocal=["01","02"]
emotion = ["01","02","03","04","05","06","07","08"]
neural = ["01"]
negative=["04","05","06","07"]
positive=["02","03","08"]
emotion_expr = ["neural","negative","positive"]
emotion_insenty = ["01","02"]
state =["01","02"]
repe =["01","02"]
actor  =["01","02","03","04","05","06","07","08","09","10","11","12","13","17","19","20","22","23"]
def save_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return
count = 0
for ac in actor:
    for r in repe:
        for s in state:
            for e_i in emotion_insenty:
                for emo in emotion:
                    for vol in vocal:
                        file=f"test/Video_Song_Actor_{ac}/Actor_{ac}/02-{vol}-{emo}-{e_i}-{s}-{r}-{ac}.mp4"
                        count +=1
                        if emo in neural:
                            save_frames(file, f"Actor_video/{ac}/neural", f'{count}')
                        elif emo in negative:
                            save_frames(file, f"Actor_video/{ac}/hukai", f'{count}')
                        elif emo in positive:
                            save_frames(file, f"Actor_video/{ac}/kai", f'{count}')
