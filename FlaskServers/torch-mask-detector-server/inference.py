from eye_detector.simple_eye_detector import eye_detector
from mask_detector.utils.nms import single_class_non_max_suppression
from mask_detector.utils.anchor_generator import generate_anchors
from mask_detector.utils.anchor_decode import decode_bbox
from mask_detector.models.torch_loader import load_torch_model, torch_inference
from bounding_box import bbox_from_two_points, convert_to_global
import numpy as np
import cv2
from nose_detector.simple_nose_detector import nose_detector

global model


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_model(model_path):
    global model
    model = load_torch_model(model_path)


def run_on_image(image,
                 conf_thresh=0.5,
                 iou_thresh=0.4,
                 ):
    """
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probability.
    :param iou_thresh: the IOU threshold of NMS
    # :param show_result: whether to display the image.
    :return:
    """
    colors = {
        "Mask": (0, 255, 0),
        "Exposed": (255, 165, 0),
        "NoMask": (255, 0, 0)
    }

    feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
    input_size = (360, 360)

    anchor_sizes = [[0.04, 0.056], [0.08, 0.11],
                    [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
    anchor_ratios = [[1, 0.62, 0.42]] * 5

    # generate anchors
    anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
    anchors_exp = np.expand_dims(anchors, axis=0)

    #### START mask detector inference ####
    height, width, _ = image.shape
    image_resized = cv2.resize(image, input_size)
    image_np = image_resized / 255.0  # normalize
    image_exp = np.expand_dims(image_np, axis=0)

    y_bboxes_output, y_cls_output = torch_inference(
        model, image_exp.transpose((0, 3, 1, 2)))

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )
    #### END mask detector inference ####

    #### START convert mask bbox, find internal nose and eye bboxes ####
    mask_boxes = []

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]

        [x1, y1, x2, y2] = bbox
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        mask_box = bbox_from_two_points(
            "Mask", x1, y1, x2, y2, (width, height))
        mask_box.set("class_id", class_id)
        mask_box.set("conf", conf)
        mask_boxes.append(mask_box)

        faceROI = mask_box.crop(image)

        #### START get potential nose and eye positions ####
        # parameter minNeighbors decides how many neighbors each candidate rectangle should have to retain it
        # larger minNeighbors -> less false positive
        nose_positions = nose_detector(faceROI, minNeighbors=2)
        nose_boxes = convert_to_global(
            "Nose", nose_positions, mask_box, width, height)
        mask_box.set("nose_boxes", nose_boxes)

        eye_positions = eye_detector(faceROI, minNeighbors=2)
        eye_boxes = convert_to_global(
            "Eye", eye_positions, mask_box, width, height)
        mask_box.set("eye_boxes", eye_boxes)
        #### END get potential nose and eye positions ####
    #### END convert mask bbox, find internal nose and eye bboxes ####
    # print(eye_boxes)
    for nose_box in nose_boxes:
        print(nose_box.center)
        if nose_box.center[0] != 251:
            image = cv2.ellipse(image, nose_box.center,
                                nose_box.halves, 0, 0, 360,
                                (0, 255, 0), 2)
            cv2.putText(image, "Nose", (nose_box.x1 + 2, nose_box.y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print(nose_box.halves)

            image = cv2.ellipse(image, nose_box.center,
                                (nose_box.halves[0]-8,
                                 nose_box.halves[1]-8), 0, 0, 360,
                                (0, 255, 0), 2)
            cv2.putText(image, "Nose", (nose_box.x1 + 2, nose_box.y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    for eye_box in eye_boxes:
        image = cv2.ellipse(image, eye_box.center,
                            eye_box.halves, 0, 0, 360,
                            (255, 0, 0), 2)
        cv2.putText(image, "Eye", (eye_box.x1 + 2, eye_box.y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    message = "No face detected."
    for mask_box in mask_boxes:
        #### START process eyes ####
        eye_boxes = mask_box.get("eye_boxes")

        # If no eyes detected, we bypass eye validation by setting the
        # threshold to the top of the image, so everything will be validated
        eye_y = max(map(lambda e: e.center[1], eye_boxes)) if len(
            eye_boxes) else 0
        #### END process eyes ####

        #### START process noses ####
        nose_boxes = mask_box.get("nose_boxes")
        largest_nose_box = max(
            nose_boxes, key=lambda nose_box: nose_box.width*nose_box.height) if len(nose_boxes) else None

        class_id = mask_box.get("class_id")
        conf = mask_box.get("conf")

        #### START analyze / draw nose presence ####
        if largest_nose_box and class_id == 0:
            largest_nose_y = largest_nose_box.center[1]
            if largest_nose_y > eye_y:
                nose_detected = True
                image = cv2.ellipse(image, largest_nose_box.center,
                                    largest_nose_box.halves, 0, 0, 360,
                                    (0, 0, 0), 2)
                cv2.putText(image, "Nose", (largest_nose_box.x1 + 2, largest_nose_box.y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                nose_detected = False
        else:
            nose_detected = False
        #### END analyze / draw nose presence ####

        #### START draw mask boundaries ####
        if class_id == 1:
            color = colors["NoMask"]
            message = "Please wear a mask!"
            text = "No Mask"
        elif nose_detected:
            color = colors["Exposed"]
            message = "Please cover you nose."
            text = "Nose Detected"
        else:
            color = colors["Mask"]
            message = "You are wearing mask properly!"
            text = "Mask"

        cv2.rectangle(image, mask_box.top_left,
                      mask_box.bottom_right, color, 2)
        cv2.putText(image, "%s: %.2f" % (text, conf), (mask_box.x1 + 2, mask_box.y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        #### END draw mask boundaries ####

    return image, message
