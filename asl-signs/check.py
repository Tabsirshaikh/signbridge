import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='holistic_landmarker.task')
options = vision.HolisticLandmarkerOptions(
    base_options=base_options
)
landmarker = vision.HolisticLandmarker.create_from_options(options)

# create a dummy blank image
import numpy as np
img = np.zeros((480, 640, 3), dtype=np.uint8)
# put a pixel
img[0,0] = [255,255,255]
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

try:
    result = landmarker.detect(mp_image)
    with open('res_struct.txt', 'w') as f:
        f.write(str(type(result.left_hand_landmarks)) + '\n')
        if len(result.left_hand_landmarks) > 0:
            f.write(str(type(result.left_hand_landmarks[0])) + '\n')
            f.write(str(dir(result.left_hand_landmarks[0])) + '\n')
        else:
            f.write("Empty left_hand_landmarks list\n")
            
        f.write(str(type(result.pose_landmarks)) + '\n')
        if len(result.pose_landmarks) > 0:
            f.write(str(type(result.pose_landmarks[0])) + '\n')
except Exception as e:
    with open('res_struct.txt', 'w') as f:
        f.write(str(e))
