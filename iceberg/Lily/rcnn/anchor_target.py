import keras.backend
import keras.engine
import tensorflow

import tensorflow_backend
import common


class AnchorTarget(keras.layers.Layer):
    """
    Calculate proposal anchor targets and corresponding labels (label: 1 is
    positive, 0 is negative, -1 is do not care) for ground truth boxes
    Arguments
        allowed_border: allow boxes to be outside the image by
        allowed_border pixels
        clobber_positives: if an anchor statisfied by positive and negative
        conditions given to negative label
        negative_overlap: IoU threshold below which labels should be given
        negative label
        positive_overlap: IoU threshold above which labels should be given
        positive label
    Input shape
        (samples, width, height, 2 * anchors), (samples, 4), (3)
    Output shape
        (samples, ), (samples, 4)
    """

    def __init__(self, allowed_border=0, clobber_positives=False,
                 negative_overlap=0.3, positive_overlap=0.7, stride=16,
                 **kwargs):
        self.allowed_border = allowed_border

        self.clobber_positives = clobber_positives

        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap

        self.stride = stride

        super(AnchorTarget, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AnchorTarget, self).build(input_shape)

    def call(self, inputs, **kwargs):
        scores, gt_boxes, metadata = inputs

        metadata = metadata[0, :]  # keras.backend.int_shape(image)[1:]

        gt_boxes = gt_boxes[0]

        rr = keras.backend.shape(scores)[1]
        cc = keras.backend.shape(scores)[2]
        total_anchors = keras.backend.shape(scores)[3]
        total_anchors = rr * cc * total_anchors

        # 1. Generate proposals from bbox deltas and shifted anchors
        all_anchors = common.shift((rr, cc), self.stride)

        # only keep anchors inside the image
        inds_inside, anchors = inside_image(all_anchors, metadata,
                                            self.allowed_border)

        # 2. obtain indices of gt boxes with the greatest overlap, balanced
        # labels
        argmax_overlaps_indices, labels = label(gt_boxes, anchors, inds_inside,
                                                self.negative_overlap,
                                                self.positive_overlap,
                                                self.clobber_positives)

        gt_boxes = keras.backend.gather(gt_boxes, argmax_overlaps_indices)

        # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
        bbox_reg_targets = common.bbox_transform(anchors, gt_boxes)

        # TODO: Why is bbox_reg_targets' shape (5, ?, 4)? Why is gt_boxes'
        # shape (None, None, 4) and not (None, 4)?
        bbox_reg_targets = keras.backend.reshape(bbox_reg_targets, (-1, 4))

        # map up to original set of anchors
        labels = unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_reg_targets = unmap(bbox_reg_targets, total_anchors, inds_inside,
                                 fill=0)

        labels = keras.backend.expand_dims(labels, axis=0)
        bbox_reg_targets = keras.backend.expand_dims(bbox_reg_targets, axis=0)
        all_anchors = keras.backend.expand_dims(all_anchors, axis=0)

        # TODO: implement inside and outside weights
        return [all_anchors, labels, bbox_reg_targets]

    def compute_output_shape(self, input_shape):
        return [(1, None, 4), (1, None), (1, None, 4)]

    def compute_mask(self, inputs, mask=None):
        # unfortunately this is required
        return 3 * [None]

    def get_config(self):
        configuration = {
            "allowed_border": self.allowed_border,
            "clobber_positives": self.clobber_positives,
            "negative_overlap": self.negative_overlap,
            "positive_overlap": self.positive_overlap,
            "stride": self.stride
        }

        return {**super(AnchorTarget, self).get_config(), **configuration}


def balance(labels):
    """
    balance labels by setting some to -1
    :param labels: array of labels (1 is positive, 0 is negative, -1 is dont
    care)
    :return: array of labels
    """

    # subsample positive labels if we have too many
    labels = subsample_positive_labels(labels)

    # subsample negative labels if we have too many
    labels = subsample_negative_labels(labels)

    return labels


def label(y_true, y_pred, inds_inside, negative_overlap=0.3,
          positive_overlap=0.7, clobber_positives=False):
    """
    Create bbox labels.
    label: 1 is positive, 0 is negative, -1 is do not care
    :param clobber_positives:
    :param positive_overlap:
    :param negative_overlap:
    :param inds_inside: indices of anchors inside image
    :param y_pred: anchors
    :param y_true: ground truth objects
    :return: indices of gt boxes with the greatest overlap, balanced labels
    """
    ones = keras.backend.ones_like(inds_inside, dtype=keras.backend.floatx())
    labels = ones * -1
    zeros = keras.backend.zeros_like(inds_inside, dtype=keras.backend.floatx())

    argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = overlapping(
        y_pred, y_true, inds_inside)

    # Assign background labels first so that positive labels can clobber them.
    if not clobber_positives:
        labels = tensorflow_backend.where(
            keras.backend.less(max_overlaps, negative_overlap), zeros, labels)

    # fg label: for each gt, anchor with highest overlap

    # TODO: generalize unique beyond 1D
    unique_indices, unique_indices_indices = tensorflow_backend.unique(
        gt_argmax_overlaps_inds, return_index=True)
    inverse_labels = keras.backend.gather(-1 * labels, unique_indices)
    unique_indices = keras.backend.expand_dims(unique_indices, 1)

    updates = keras.backend.ones_like(
        keras.backend.reshape(unique_indices, (-1,)),
        dtype=keras.backend.floatx())
    labels = tensorflow_backend.scatter_add_tensor(labels, unique_indices,
                                                   inverse_labels + updates)

    # Assign foreground labels based on IoU overlaps that are higher than
    # RPN_POSITIVE_OVERLAP.
    labels = tensorflow_backend.where(
        keras.backend.greater_equal(max_overlaps, positive_overlap), ones,
        labels)

    if clobber_positives:
        # assign bg labels last so that negative labels can clobber positives
        labels = tensorflow_backend.where(
            keras.backend.less(max_overlaps, negative_overlap), zeros, labels)

    return argmax_overlaps_inds, balance(labels)


def overlapping(anchors, gt_boxes, inds_inside):
    """
    overlaps between the anchors and the gt boxes
    :param anchors: Generated anchors
    :param gt_boxes: Ground truth bounding boxes
    :param inds_inside:
    :return:
    """

    assert keras.backend.ndim(anchors) == 2
    assert keras.backend.ndim(gt_boxes) == 2

    reference = common.overlap(anchors, gt_boxes)

    gt_argmax_overlaps_inds = keras.backend.argmax(reference, axis=0)

    argmax_overlaps_inds = keras.backend.argmax(reference, axis=1)

    arranged = keras.backend.arange(0, keras.backend.shape(inds_inside)[0])

    indices = keras.backend.stack(
        [arranged, keras.backend.cast(argmax_overlaps_inds, "int32")], axis=0)

    indices = keras.backend.transpose(indices)

    max_overlaps = tensorflow_backend.gather_nd(reference, indices)

    return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds


def subsample_negative_labels(labels, rpn_batchsize=256):
    """
    subsample negative labels if we have too many
    :param labels: array of labels (1 is positive, 0 is negative, -1 is dont
    care)
    :return:
    """
    num_bg = rpn_batchsize - keras.backend.shape(
        tensorflow_backend.where(keras.backend.equal(labels, 1)))[0]

    bg_inds = tensorflow_backend.where(keras.backend.equal(labels, 0))

    num_bg_inds = keras.backend.shape(bg_inds)[0]

    size = num_bg_inds - num_bg

    def more_negative():
        indices = keras.backend.reshape(bg_inds, (-1,))
        indices = tensorflow_backend.shuffle(indices)[:size]

        updates = tensorflow.ones((size,)) * -1

        inverse_labels = keras.backend.gather(labels, indices) * -1

        indices = keras.backend.reshape(indices, (-1, 1))

        return tensorflow_backend.scatter_add_tensor(labels, indices,
                                                     inverse_labels + updates)

    condition = keras.backend.less_equal(size, 0)

    return keras.backend.switch(condition, labels, lambda: more_negative())


def subsample_positive_labels(labels, rpn_fg_fraction=0.5, rpn_batchsize=256):
    """
    subsample positive labels if we have too many
    :param labels: array of labels (1 is positive, 0 is negative,
    -1 is dont care)
    :return:
    """

    num_fg = int(rpn_fg_fraction * rpn_batchsize)

    fg_inds = tensorflow_backend.where(keras.backend.equal(labels, 1))
    num_fg_inds = keras.backend.shape(fg_inds)[0]

    size = num_fg_inds - num_fg

    def more_positive():
        indices = keras.backend.reshape(fg_inds, (-1,))
        indices = tensorflow_backend.shuffle(indices)[:size]

        updates = tensorflow.ones((size,)) * -1

        inverse_labels = keras.backend.gather(labels, indices) * -1

        indices = keras.backend.reshape(indices, (-1, 1))

        updates = inverse_labels + updates

        return tensorflow_backend.scatter_add_tensor(labels, indices, updates)

    condition = keras.backend.less_equal(size, 0)

    return keras.backend.switch(condition, labels, lambda: more_positive())


def unmap(data, count, inds_inside, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if keras.backend.ndim(data) == 1:
        ret = tensorflow.ones((count,), dtype=keras.backend.floatx()) * fill

        inds_nd = keras.backend.expand_dims(inds_inside)
    else:
        ret = (count, keras.backend.shape(data)[1])
        ret = tensorflow.ones(ret, dtype=keras.backend.floatx()) * fill

        data = keras.backend.transpose(data)
        data = keras.backend.reshape(data, (-1,))

        inds_ii = keras.backend.tile(inds_inside, [4])
        inds_ii = keras.backend.expand_dims(inds_ii)

        ones = keras.backend.expand_dims(keras.backend.ones_like(inds_inside),
                                         1)

        inds_coords = keras.backend.concatenate(
            [ones * 0, ones, ones * 2, ones * 3], 0)

        inds_nd = keras.backend.concatenate([inds_ii, inds_coords], 1)

    inverse_ret = tensorflow_backend.gather_nd(-1 * ret, inds_nd)
    inverse_ret = tensorflow_backend.squeeze(inverse_ret)

    updates = inverse_ret + data
    ret = tensorflow_backend.scatter_add_tensor(ret, inds_nd, updates)

    return ret


def inside_image(boxes, im_info, allowed_border=0):
    """
    Calc indices of boxes which are located completely inside of the image
    whose size is specified by img_info ((height, width, scale)-shaped array).
    :param boxes: (None, 4) tensor containing boxes in original image
    (x1, y1, x2, y2)
    :param im_info: (height, width, scale)
    :param allowed_border: allow boxes to be outside the image by
    allowed_border pixels
    :return: (None, 4) indices of boxes completely in original image, (None,
    4) tensor of boxes completely inside image
    """

    indices = tensorflow_backend.where(
        (boxes[:, 0] >= -allowed_border) &
        (boxes[:, 1] >= -allowed_border) &
        (boxes[:, 2] < allowed_border + im_info[1]) &  # width
        (boxes[:, 3] < allowed_border + im_info[0])  # height
    )

    indices = keras.backend.cast(indices, "int32")

    gathered = keras.backend.gather(boxes, indices)

    return indices[:, 0], keras.backend.reshape(gathered, [-1, 4])


def inside_and_outside_weights(anchors, subsample, positive_weight, proposed_inside_weights):
    """
    Creates the inside_weights and outside_weights bounding-box weights.
    Args:
        anchors: Generated anchors.
        subsample:  Labels obtained after subsampling.
        positive_weight:
        proposed_inside_weights:
    Returns:
        inside_weights:  Inside bounding-box weights.
        outside_weights: Outside bounding-box weights.
    """
    number_of_anchors = keras.backend.int_shape(anchors)[0]

    proposed_inside_weights = keras.backend.constant([proposed_inside_weights])
    proposed_inside_weights = keras.backend.tile(proposed_inside_weights, (number_of_anchors, 1))

    positive_condition = keras.backend.equal(subsample, 1)
    negative_condition = keras.backend.equal(subsample, 0)

    if positive_weight < 0:
        # Assign equal weights to both positive_weights and negative_weights
        # labels.
        examples = keras.backend.cast(negative_condition, keras.backend.floatx())
        examples = keras.backend.sum(examples)

        positive_weights = keras.backend.ones_like(anchors) / examples
        negative_weights = keras.backend.ones_like(anchors) / examples
    else:
        # Assign weights that favor either the positive or the
        # negative_weights labels.
        assert (positive_weight > 0) & (positive_weight < 1)

        positive_examples = keras.backend.cast(positive_condition, keras.backend.floatx())
        positive_examples = keras.backend.sum(positive_examples)

        negative_examples = keras.backend.cast(negative_condition, keras.backend.floatx())
        negative_examples = keras.backend.sum(negative_examples)

        positive_weights = keras.backend.ones_like(anchors) * (0 + positive_weight) / positive_examples
        negative_weights = keras.backend.ones_like(anchors) * (1 - positive_weight) / negative_examples

    inside_weights = keras.backend.zeros_like(anchors)
    inside_weights = tensorflow_backend.where(positive_condition, proposed_inside_weights, inside_weights)

    outside_weights = keras.backend.zeros_like(anchors)
    outside_weights = tensorflow_backend.where(positive_condition, positive_weights, outside_weights)
    outside_weights = tensorflow_backend.where(negative_condition, negative_weights, outside_weights)

    return inside_weights, outside_weights