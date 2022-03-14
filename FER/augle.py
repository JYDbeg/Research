#!/usr/bin/env python
from mlxtend.image import extract_face_landmarks
import cv2
import numpy as np

# Read Image
def euler(n):
    im = cv2.imread(n);
    size = im.shape
    landmarks = extract_face_landmarks(im)
    #2D image points. If you change the image, you need to change vector
    i_p = np.array([30,8,36,45,48,54])
    image_points =  np.float32([landmarks[i_p]])
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner

                            ])

    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )


    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    x = euler_angle[0]
    y = euler_angle[1]
    z = euler_angle[2]
    return x,y,z