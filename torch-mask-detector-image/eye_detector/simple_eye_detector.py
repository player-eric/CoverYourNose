import cv2 as cv

eyes_cascade_file = "eye_detector/cascade/haarcascade_eye_tree_eyeglasses.xml"
eyes_cascade = cv.CascadeClassifier()

if not eyes_cascade.load(eyes_cascade_file):
    print('--(!)Error loading eyes cascade')
    exit(0)


def eye_detector(image, minNeighbors=8):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.equalizeHist(image)

    # detect eyes in the input image
    eyes = eyes_cascade.detectMultiScale(image, minNeighbors=minNeighbors)

    return eyes
