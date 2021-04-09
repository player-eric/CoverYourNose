import cv2

import numpy as np
#from keras.models import model_from_json
from mask_detector.utils.anchor_generator import generate_anchors
from mask_detector.utils.anchor_decode import decode_bbox
from mask_detector.utils.nms import single_class_non_max_suppression
from mask_detector.models.tensorflow_loader import load_tf_model, tf_inference


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
              draw_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param input_size: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    # :param show_result: whether to display the image.
    :return:
    '''

    #### START get potential nose positions before passing the image into mask detector ####
    #### parameter minNeighbors decides how many neighbors each candidate rectangle should have to retain it
    #### larger minNeighbors -> less false positive
    #### This simple nose detector is not working perfectly, I'll try to improve it later
    nose_positions = nose_detector(image, minNeighbors = 5)
    #### END get potential nose positions before passing the image into mask detector ####

    #### START nose detector inference ####
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, input_size)
    image_np = image_resized / 255.0  # normalize
    image_exp = np.expand_dims(image_np, axis = 0)
    
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
    #### END nose detector inference ####

    #### START draw mask boundary ####
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
    #### END draw mask boundary ####
    
    #### START draw noses ####
    #print(nose_positions)
    for (x, y, w, h) in nose_positions:
        center = (x + w//2, y + h//2)
        image = cv2.ellipse(image, center, (w // 2, h // 2), 0, 0, 360, (0, 0, 0), 4)
        cv2.putText(image, "Nose", (x + 2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    #### END draw noses ####

    
    return output_info, image
