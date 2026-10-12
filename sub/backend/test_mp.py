import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os

task_path = r"c:\yaseer code\sem 4\sign language\New folder (2)\holistic_landmarker.task"
print("Loading model from", task_path)

base_opts = mp_python.BaseOptions(model_asset_path=task_path)
mp_opts   = vision.HolisticLandmarkerOptions(base_options=base_opts)
holistic  = vision.HolisticLandmarker.create_from_options(mp_opts)

# Create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw something so it might find a shape (doesn't matter if it fails to find a person, we just want to see the object structure)
# Actually, an all-black image will result in empty landmarks. We want to see what happens when it finds a person.
# But even if we don't find a person, we can just print the type of the result object.
print("Detecting...")
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
res = holistic.detect(mp_img)

print("pose_landmarks:", type(res.pose_landmarks))
if res.pose_landmarks:
    print("pose_landmarks[0]:", type(res.pose_landmarks[0]))
    if hasattr(res.pose_landmarks[0], '__iter__'):
        print("pose_landmarks[0][0]:", type(list(res.pose_landmarks[0])[0]))

print("Done.")
