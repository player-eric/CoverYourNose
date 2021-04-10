import cv2
import numpy as np

from bounding_box import bbox_from_two_points
from eye_detector.simple_eye_detector import eye_detector
from mask_detector.models.tensorflow_loader import load_tf_model, tf_inference
from mask_detector.utils.anchor_decode import decode_bbox
from mask_detector.utils.anchor_generator import generate_anchors
from mask_detector.utils.nms import single_class_non_max_suppression

# import matplotlib.pyplot as plt

# from keras.models import model_from_json

#### imports for a CNN mask detector start ####
#### these can be later replaced by model trained by ourselves ####

sess, graph = load_tf_model('mask_detector/models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)
id2class = {0: 'Mask', 1: 'NoMask'}

#### imports for a CNN mask detector end ####

#### imports for a HAAR cascade based nose detector start ####
from nose_detector.simple_nose_detector import nose_detector


#### imports for a HAAR cascade based nose detector end ####

def run_on_image(image,
                 conf_thresh=0.5,
                 iou_thresh=0.4,
                 input_size=(160, 160),
                 draw_result=True,
                 ):
    """
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probability.
    :param iou_thresh: the IOU threshold of NMS
    :param input_size: the model input size.
    :param draw_result: whether to draw bounding box to the image.
    # :param show_result: whether to display the image.
    :return:
    """
    #### START mask detector inference ####
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, input_size)
    image_np = image_resized / 255.0  # normalize
    image_exp = np.expand_dims(image_np, axis=0)

    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

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

    #### START compute mask boundary ####
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

        m_box = bbox_from_two_points("Mask", x1, y1, x2, y2, (width, height))
        m_box.set("class_id", class_id)
        m_box.set("conf", conf)
        mask_boxes.append(m_box)

        # plt.imshow(m_box.crop(image))
        # plt.show()

        localize = m_box, width, height
        #### START get potential nose and eye positions ####
        #### parameter minNeighbors decides how many neighbors each candidate rectangle should have to retain it
        #### larger minNeighbors -> less false positive
        #### This simple nose detector is not working perfectly, I'll try to improve it later
        m_box.set("nose_boxes", nose_detector(image, minNeighbors=10, localize=localize))
        m_box.set("eye_boxes", eye_detector(image, minNeighbors=8, localize=localize))
        #### END get potential nose and eye positions ####

        output_info.append([class_id, conf, *m_box.top_left, *m_box.bottom_right])
    #### END compute mask boundary ####

    #### END draw mask boundaries ####
    for m_box in mask_boxes:
        eye_boxes = m_box.get("eye_boxes")

        assert len(eye_boxes) <= 2

        # if draw_result:
        #     for e_box in eye_boxes:
        #         radius = int(round((e_box.width + e_box.height) * 0.25))
        #         image = cv2.circle(image, e_box.center, radius, (0, 0, 0), 2)

        eye_y = max(eye_boxes, key=lambda e: e.center[1]).center[1] if len(eye_boxes) else 0

        print(f"Eye Y Threshold: {eye_y}")

        nose_boxes = m_box.get("nose_boxes")
        validated_noses = 0

        if not len(nose_boxes):
            print("No noses detected.")

        for n_box in nose_boxes:
            nose_y = n_box.center[1]
            print(f"Nose Box Center Y: {nose_y}")
            if nose_y > eye_y:
                print("Validated.")
                validated_noses += 1
                if draw_result:
                    image = cv2.ellipse(image, n_box.center, n_box.halves, 0, 0, 360, (0, 0, 0), 4)
                    cv2.putText(image, "Nose", (n_box.x1 + 2, n_box.y1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                print("Rejected.")

        contains_nose = validated_noses > 0
        class_id = m_box.get("class_id")
        contains_mask = class_id == 0

        text = id2class[class_id]
        if not contains_mask:
            color = (255, 0, 0)
        elif contains_nose:
            text = "+Nose"
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(image, m_box.top_left, m_box.bottom_right, color, 2)
        cv2.putText(image, "%s: %.2f" % (text, m_box.get("conf")), (m_box.x1 + 2, m_box.y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
    #### END draw mask boundaries ####

    return output_info, image
