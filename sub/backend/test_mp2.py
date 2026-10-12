import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os

task_path = r"c:\yaseer code\sem 4\sign language\New folder (2)\holistic_landmarker.task"

base_opts = mp_python.BaseOptions(model_asset_path=task_path)
mp_opts   = vision.HolisticLandmarkerOptions(base_options=base_opts)
holistic  = vision.HolisticLandmarker.create_from_options(mp_opts)

# Create a dummy image with something that looks like a person
# We can just run it on a real image if we have one. But let's just make a white line so it might detect a pose.
# Actually, the quickest way to see the true structure is to just read the source docs or instantiate an object.
print("Holistic Result:", dir(vision.HolisticLandmarkerResult))

img = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.circle(img, (320, 100), 40, (0,0,0), -1) # head
cv2.line(img, (320, 140), (320, 300), (0,0,0), 10) # body
cv2.line(img, (320, 180), (200, 250), (0,0,0), 10) # arms
cv2.line(img, (320, 180), (440, 250), (0,0,0), 10) 

mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
res = holistic.detect(mp_img)

if res.pose_landmarks:
    print("POSE Is list?", isinstance(res.pose_landmarks, list))
    print("POSE[0] Is list?", isinstance(res.pose_landmarks[0], list))
    if isinstance(res.pose_landmarks[0], list) and len(res.pose_landmarks[0]) > 0:
        print("POSE[0][0] type:", type(res.pose_landmarks[0][0]))
else:
    print("No pose detected to verify structure.")
