from __future__ import print_function
import cv2 as cv
import argparse
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    eyes = eyes_cascade.detectMultiScale(frame_gray, minNeighbors=2)
    for (x, y, w, h) in eyes:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (0, 0, 255), 4)
    
    nose = nose_cascade.detectMultiScale3(frame_gray, minNeighbors=8, outputRejectLevels=True)
    #print(nose)
    for (x, y, w, h) in nose[0]:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (0, 0, 0), 4)
        
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray,minNeighbors=1)
    for (x, y, w, h) in faces:
        
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect nose
        nose = nose_cascade.detectMultiScale(faceROI, minNeighbors=4)
        if len(nose) > 0:
            center = (x + w//2, y + h//2)
            frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            
        #nose_cascade.detectMultiScale3(faceROI,outputRejectLevels=True)
        for (x2,y2,w2,h2) in nose:
            nose_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, nose_center, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--nose_cascade', help='Path to nose cascade.', default='data/haarcascades/haarcascade_nose_tree_noseglasses.xml')
parser.add_argument('--eyes_cascade', help='Path to eye cascade.', default='./haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
nose_cascade_name = args.nose_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
nose_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(face_cascade_name):
    print('--(!)Error loading face cascade')
    exit(0)
if not nose_cascade.load(nose_cascade_name):
    print('--(!)Error loading nose cascade')
    exit(0)
if not eyes_cascade.load(eyes_cascade_name):
    print('--(!)Error loading nose cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break
