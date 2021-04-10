import cv2 as cv

from bounding_box import bbox_from_anchor_dims

nose_cascade_file = "nose_detector/cascade/nose25x15.xml"
nose_cascade = cv.CascadeClassifier()
if not nose_cascade.load(nose_cascade_file):
    print('--(!)Error loading nose cascade')
    exit(0)


def nose_detector(image, minNeighbors=8, localize=None):
    if localize is not None:
        image = localize[0].crop(image)

    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.equalizeHist(image)

    # detect noses in the input image
    nose = nose_cascade.detectMultiScale3(image, minNeighbors=minNeighbors, outputRejectLevels=True)
    nose_positions = nose[0]

    if localize is not None:
        mask_box, width, height = localize
        n_boxes = []
        for (x, y, w, h) in nose_positions:
            n_box = bbox_from_anchor_dims("Nose", x, y, w, h, (width, height))
            mask_box.to_global_coordinates(n_box)
            assert mask_box.contains(n_box)
            n_boxes.append(n_box)
        return n_boxes

    return nose_positions
