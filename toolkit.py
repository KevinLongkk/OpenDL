import toolkit as tools
import tensorflow as tf


"""
    Non-max Suppression Algorithm

    @param list  Object candidate bounding boxes
    @param list  Confidence score of bounding boxes
    @param float IoU threshold

    @return Rest boxes after nms operation
"""
def nms(dets, threshold):
    return tools.nms.nms(dets, threshold)


"""
    cal_iou
    
    @:param predictions_boxes
"""
def cal_iou(predictions_boxes, labels_boxes, num_grids, predict_per_cell=1, num_anchors=1):
    offset_x = tf.constant([x / num_grids for x in range(num_grids)] * num_grids, dtype=tf.float32)
    offset_x = tf.reshape(offset_x, (1, num_grids, num_grids))
    offset_x = tf.reshape(tf.tile(offset_x, [1, 1, predict_per_cell]),
                          (1, num_grids, num_grids, predict_per_cell))
    offset_x = tf.reshape(tf.tile(offset_x, [1, 1, 1, num_anchors]),
                          (1, num_grids, num_grids, predict_per_cell, num_anchors))

    offset_y = tf.transpose(offset_x, (0, 2, 1, 3, 4))

    labels_offset = tf.stack([
        labels_boxes[..., 0] / num_grids + offset_x,
        labels_boxes[..., 1] / num_grids + offset_y,
        labels_boxes[..., 2],
        labels_boxes[..., 3]
    ], axis=-1)

    predictions_offset = tf.stack([
        predictions_boxes[..., 0] / num_grids + offset_x,
        predictions_boxes[..., 1] / num_grids + offset_y,
        predictions_boxes[..., 2],
        predictions_boxes[..., 3]
    ], axis=-1)

    xmin = tf.maximum(labels_offset[..., 0] - labels_offset[..., 2] / 2,
                      predictions_offset[..., 0] - predictions_offset[..., 2] / 2)
    ymin = tf.maximum(labels_offset[..., 1] - labels_offset[..., 3] / 2,
                      predictions_offset[..., 1] - predictions_offset[..., 3] / 2)
    xmax = tf.minimum(labels_offset[..., 0] + labels_offset[..., 2] / 2,
                      predictions_offset[..., 0] + predictions_offset[..., 2] / 2, )
    ymax = tf.minimum(labels_offset[..., 1] + labels_offset[..., 3] / 2,
                      predictions_offset[..., 1] + predictions_offset[..., 3] / 2)
    intersection = tf.maximum(0.0, xmax - xmin) * tf.maximum(0.0, ymax - ymin)
    union = predictions_boxes[..., 2] * predictions_boxes[..., 3] + \
            labels_boxes[..., 2] * labels_boxes[..., 3] - intersection
    union = tf.maximum(union, 1e-10)
    return intersection / union






