import numpy as np

from utils.classbox import BoundBox
from utils.loadimage import resize_and_scale
from detection.model import get_model


def sigmoid(x):
    """
    Sigmoid function
    :param x:
    :return : sigmoid of x
    """
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, img_h, img_w, nb_box=3):
    """
    This function decodes netout for one grid size
    :param netout:          output from on size detection
    :param anchors:         set of predefined bounding boxes
    :param obj_thresh:      IOU  threshold
    :param img_h:           height of image
    :param img_w:           width  of image
    :param nb_box:          the number of predefined bounding boxes

    :return: ndarray with predicted boxes. size = (number of boxes, 85)
    """

    grid_h, grid_w = netout.shape[:2]
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    netout[..., :2] = sigmoid(netout[..., :2])
    netout[..., 4:] = sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    # array with columns numbers
    _columns_array = np.arange(grid_h).repeat(grid_h * nb_box)
    _columns_array = _columns_array.reshape((grid_h, grid_w, nb_box, 1))

    # array with rows numbers
    _rows_array = np.arange(grid_w)
    _rows_array = np.repeat(_rows_array, nb_box, axis=0)
    _rows_array = np.tile(_rows_array, (1, grid_w)).reshape((grid_h, grid_w, nb_box, 1))

    # convert x, y, w, h relative to image size
    y_arr = netout[..., 1:2]
    y_arr = (y_arr + _columns_array) / grid_h

    x_arr = netout[..., 0:1]
    x_arr = (x_arr + _rows_array) / grid_w

    w_and_h = netout[..., 2:4]
    w_and_h = np.exp(w_and_h) * np.array(anchors).reshape((nb_box, 2)) / img_w

    # concatenate coordinates subject score and classes score
    boxes = np.concatenate((x_arr, y_arr, w_and_h, netout[..., 4:]), axis=3)
    boxes = boxes[(boxes[..., 4:5] > obj_thresh).all(axis=3)]

    return boxes


def convert_coordinates_to_minmax(boxes):
    """
    convert coordinates x, y, w, h to x_min, y_min, x_max, y_max relative to image size
    0 <= x_min, y_min, x_max, y_max <= 1

    :param boxes: ndarray with size = (number of boxes, 85)

    :return: ndarray with size = (number of boxes, 85)
    """
    # x, y, w, h
    # box = x_min, y_min, x_max, y_max = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    x = boxes[..., 0:1]
    y = boxes[..., 1:2]
    w = boxes[..., 2:3]
    h = boxes[..., 3:4]

    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2

    return np.concatenate((x_min, y_min, x_max, y_max, boxes[..., 4:]), axis=1)


def resize_to_bild_size(boxes, image_size):
    """
    resize the relative coordinate of the boxes to imagesize
    :param boxes: ndarray with size = (number of boxes, 85)
    :param image_size: tuple of width, height of image

    :return: ndarray with size = (number of boxes, 85)
    """
    width, height = image_size
    boxes[..., 0] = boxes[..., 0] * width
    boxes[..., 1] = boxes[..., 1] * height
    boxes[..., 2] = boxes[..., 2] * width
    boxes[..., 3] = boxes[..., 3] * height

    boxes[..., 0:4] = boxes[..., 0:4].astype(int)

    return boxes


def bbox_iou(box1, box2):
    """

    :param box1: object from class BoundBox
    :param box2: object from class BoundBox
    :return  IOU -> float:
    """
    x1 = max(box1.xmin, box2.xmin)
    x2 = min(box1.xmax, box2.xmax)
    y1 = max(box1.ymin, box2.ymin)
    y2 = min(box1.ymax, box2.ymax)

    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    if intersect == 0:
        return 0

    box1Area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    box2Area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)

    return intersect / float(box1Area + box2Area - intersect)


def do_nms(boxes, nms_thresh, ):
    # temp list to save boxes after NMS
    clear_boxes = []

    # check of any boxes are found  with high confidence score
    if len(boxes) == 0:
        pass

    # select columns with existing classes
    exist_classes = np.where(np.any(boxes[..., 5:] != 0, axis=0) == True)[0] + 5

    # for each class do NMS
    for c in exist_classes:

        # select boxes with current class
        selected_boxes = boxes[(boxes[:, c] > 0)]

        # Compute the area of the bounding boxes
        #  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        areas = (selected_boxes[:, 2] - selected_boxes[:, 0] + 1) * (selected_boxes[:, 3] - selected_boxes[:, 1] + 1)

        # sort indexes from height to low confident score
        sort_indexes = np.flip(np.argsort(selected_boxes[:, 4]))

        while sort_indexes.size != 0:
            # box with
            current_box_index = sort_indexes[0]
            clear_boxes.append(selected_boxes[sort_indexes[0]])
            sort_indexes = sort_indexes[1:]

            xx1 = np.maximum(selected_boxes[current_box_index][0], selected_boxes[sort_indexes, 0])
            yy1 = np.maximum(selected_boxes[current_box_index][1], selected_boxes[sort_indexes, 1])
            xx2 = np.minimum(selected_boxes[current_box_index][2], selected_boxes[sort_indexes, 2])
            yy2 = np.minimum(selected_boxes[current_box_index][3], selected_boxes[sort_indexes, 3])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / areas[sort_indexes]

            sort_indexes = np.delete(sort_indexes, np.argwhere(overlap > nms_thresh))

    return clear_boxes


def convert_to_BoundBox_class(boxes):
    converted_boxes = []
    for _ in boxes:
        objectness = _[4]
        classes = _[5:]
        box = BoundBox(_[0], _[1], _[2], _[3], objectness, classes)
        converted_boxes.append(box)

    return converted_boxes


model = get_model(file_name="weights/model.h5")


def predict_boxes(image, class_threshold, nms_thresh, ANCHORS, IMG_SIZE):
    input_w, input_h = IMG_SIZE

    # load and prepare image
    # image = load_image_pixels(image)
    image_w, image_h = image.size
    image = resize_and_scale(image, (input_w, input_h))

    # make prediction
    yhat = model.predict(image)

    # define the probability threshold for detected objects
    boxes = np.empty([0, 85])
    for i in range(len(yhat)):
        # decode the output of the network
        b = decode_netout(yhat[i][0], ANCHORS[i], class_threshold, input_h, input_w)
        boxes = np.concatenate([boxes, b])

    # correct the sizes of the bounding boxes for the shape of the image
    boxes = convert_coordinates_to_minmax(boxes)
    boxes = resize_to_bild_size(boxes, (image_w, image_h))

    # do NMS for all predicted boxes
    boxes = do_nms(boxes, nms_thresh)

    # convert ndarray with clear boxes to list with boxes as BoundBox class
    boxes = convert_to_BoundBox_class(boxes)

    return boxes
