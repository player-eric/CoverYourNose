import cv2 as cv

from bounding_box import bbox_from_anchor_dims

eyes_cascade_file = "eye_detector/cascade/haarcascade_eye_tree_eyeglasses.xml"

eyes_cascade = cv.CascadeClassifier()

if not eyes_cascade.load(eyes_cascade_file):
    print('--(!)Error loading eyes cascade')
    exit(0)


def eye_detector(image, minNeighbors=8, localize=None):
    if localize is not None:
        image = localize[0].crop(image)

    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.equalizeHist(image)

    # detect eyes in the input image
    eyes = eyes_cascade.detectMultiScale(image, minNeighbors=minNeighbors)

    if localize is not None:
        mask_box, width, height = localize
        n_boxes = []
        for (x, y, w, h) in eyes:
            n_box = bbox_from_anchor_dims("Eye", x, y, w, h, (width, height))
            mask_box.to_global_coordinates(n_box)
            assert mask_box.contains(n_box)
            n_boxes.append(n_box)
        return n_boxes

    return eyes
