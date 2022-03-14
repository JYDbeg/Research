import subprocess
import numpy as np
id_name = np.arange(1,41,1)
for i in id_name:
    print(str(i).zfill(3))
    
root_path = "D:/BaiduNetdiskDownload/py/OpenFace_2.2.0_win_x64/FaceGrabberDB/"
emotion=["01_HeadScan","02_Anger","03_Disgust","04_Fear","05_Happiness","06_Sadness","07_Surprise","08_Random"]
subdir = "KinectColorRaw"
for i in id_name:
    for k in emotion:
        file_dir = root_path + "ID_" + str(i).zfill(3)+"/"+k+"/"+subdir
        subprocess.call(["FeatureExtraction.exe","-fdir",file_dir,"-out_dir",file_dir+"/result"])
    