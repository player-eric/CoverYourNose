import cv2 as cv

nose_cascade_file = "nose_detector/cascade/nose25x15.xml"
nose_cascade = cv.CascadeClassifier()

if not nose_cascade.load(nose_cascade_file):
    print('--(!)Error loading nose cascade')
    exit(0)


def nose_detector(image, minNeighbors=8):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.equalizeHist(image)

    # detect noses in the input image
    noses = nose_cascade.detectMultiScale3(image, minNeighbors=minNeighbors, outputRejectLevels=True)[0]

    return noses
