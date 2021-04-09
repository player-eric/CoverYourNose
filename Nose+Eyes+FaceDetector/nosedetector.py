
# run with command:
# python nosedetector.py nose.jpg
# press 0 to close the result

import sys
import cv2
# Get user supplied values
imagePath = sys.argv[1]
noseCascadePath = "./nose18x15.xml"
# Create the haar cascade

noseCascade = cv2.CascadeClassifier(noseCascadePath)
# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Should draw a rectangle around the faces, temporarily use the whole image
h , w , _ = image.shape
y = 0
x = 0
roi_gray = gray[y:y+h, x:x+w]
roi_color = image[y:y+h, x:x+w]
nose = noseCascade.detectMultiScale(roi_gray)
print(nose)
for (ex,ey,ew,eh) in nose:
    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
cv2.imshow("Nose found" ,image)
cv2.waitKey(0) 
