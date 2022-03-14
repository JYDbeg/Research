import tensorflow as tf
import config
import pathlib
from config import image_height, image_width, channels
import mediapipe as mp
import numpy as np
import cv2
import face_recognition
from tqdm import tqdm
def load_and_preprocess_image(img_path):
    # read pictures
    image = cv2.imread(img_path)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                face_location = face_recognition.face_locations(image,0,"cnn")
                for face in face_location:
                    top,right,bottom,left = face
                    image = image[top:bottom,left:right]
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5) as face_mesh:
                    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(image, face_landmarks,mp_face_mesh.FACEMESH_CONTOURS,None,mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                        poss=[]
                        for lm in face_landmarks.landmark:
                            poss.append([lm.x,lm.y,lm.z])
                        poss = np.array(poss)
                        listdata = []
                        poss_mean = np.mean(poss,axis =0)
                        for i in poss:
                            listdata.append([np.sqrt((i[0]-poss_mean[0])**2+(i[1]-poss_mean[1])**2+
                                                        (i[2]-poss_mean[2])**2),np.arccos((i[0]*poss_mean[0]+i[1]*poss_mean[1]+i[2]*poss_mean[2])/(np.sqrt(i[0]**2+i[1]**2+i[2]**2)
                                                                                                                                                *np.sqrt(poss_mean[0]**2+poss_mean[1]**2+poss_mean[2]**2)))])
                        listdata = np.array(listdata)
                        listdata = listdata.flatten()
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks,mp_face_mesh.FACEMESH_CONTOURS,None,mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
                poss=[]
                for lm in face_landmarks.landmark:
                    poss.append([lm.x,lm.y,lm.z])
                poss = np.array(poss)
                listdata = []
                poss_mean = np.mean(poss,axis =0)
                for i in poss:
                    listdata.append([np.sqrt((i[0]-poss_mean[0])**2+(i[1]-poss_mean[1])**2+
                                                (i[2]-poss_mean[2])**2),np.arccos((i[0]*poss_mean[0]+i[1]*poss_mean[1]+i[2]*poss_mean[2])/(np.sqrt(i[0]**2+i[1]**2+i[2]**2)
                                                                                                                                           *np.sqrt(poss_mean[0]**2+poss_mean[1]**2+poss_mean[2]**2)))])
                listdata = np.array(listdata)
                listdata = listdata.flatten()
                
    except:
        return 0
    # decode pictures
    # resize
    # normalization
    #img = img_tensor / 255.0
    return listdata

def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_dataset(dataset_root_dir):
    all_image_path, all_image_label = get_images_and_labels(data_root_dir=dataset_root_dir)
    # print("image_path: {}".format(all_image_path[:]))
    # print("image_label: {}".format(all_image_label[:]))
    # load the dataset and preprocess images
    all_list = []
    for idx,image_path in tqdm(enumerate(all_image_path)):
        if type(load_and_preprocess_image(image_path)) is int:
            all_image_label.pop(idx)
        else:
            all_list.append(load_and_preprocess_image(image_path))
    image_dataset = tf.data.Dataset.from_tensor_slices(all_list)
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    image_count = len(all_image_path)

    return dataset, image_count


def generate_datasets():
    train_dataset, train_count = get_dataset(dataset_root_dir=config.train_dir)
    valid_dataset, valid_count = get_dataset(dataset_root_dir=config.valid_dir)
    test_dataset, test_count = get_dataset(dataset_root_dir=config.test_dir)


    # read the original_dataset in the form of batch
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count

