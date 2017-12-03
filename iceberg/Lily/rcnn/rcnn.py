import keras.backend
import keras.engine
import keras.layers

import pooling
import anchor_target
import object_proposal
import proposal_target
import rpn_loss
import rcnn_loss
import object_detection


def _extract_regions(classes):
    def f(inputs):
        features, metadata, proposals = inputs

        regions = pooling.RegionOfInterest(extent=(14, 14))([features, proposals, metadata])

        regions = keras.layers.TimeDistributed(keras.layers.Flatten())(regions)

        regions = keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"))(regions)
        regions = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))(regions)

        regions = keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"))(regions)
        regions = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))(regions)

        deltas = keras.layers.TimeDistributed(keras.layers.Dense(4 * classes, activation="linear", kernel_initializer="zero"))(regions)
        scores = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax", kernel_initializer="zero"))(regions)

        return [deltas, scores]

    return f


class RCNN(keras.models.Model):
    def __init__(self, image, classes):
        inputs = [
            keras.layers.Input((None, 4)),
            image,
            keras.layers.Input((None, classes)),
            keras.layers.Input((3,))
        ]

        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        bounding_boxes, image, labels, metadata = inputs

        features = keras.layers.Conv2D(64, name="convolution_1_1", **options)(image)
        features = keras.layers.Conv2D(64, name="convolution_1_2", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_1")(features)

        features = keras.layers.Conv2D(128, name="convolution_2_1", **options)(features)
        features = keras.layers.Conv2D(128, name="convolution_2_2", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_2")(features)

        features = keras.layers.Conv2D(256, name="convolution_3_1", **options)(features)
        features = keras.layers.Conv2D(256, name="convolution_3_2", **options)(features)
        features = keras.layers.Conv2D(256, name="convolution_3_3", **options)(features)
        features = keras.layers.Conv2D(256, name="convolution_3_4", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_3")(features)

        features = keras.layers.Conv2D(512, name="convolution_4_1", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_4_2", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_4_3", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_4_4", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_4")(features)

        features = keras.layers.Conv2D(512, name="convolution_5_1", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_5_2", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_5_3", **options)(features)

        convolution_3x3 = keras.layers.Conv2D(512, name="convolution_3x3", **options)(features)

        deltas = keras.layers.Conv2D(9 * 4, (1, 1), activation="linear", kernel_initializer="zero", name="deltas")(convolution_3x3)
        scores = keras.layers.Conv2D(9 * 1, (1, 1), activation="sigmoid", kernel_initializer="uniform", name="scores")(convolution_3x3)

        anchors, rpn_labels, bounding_box_targets = anchor_target.AnchorTarget()([scores, bounding_boxes, metadata])

        deltas = rpn_loss.RPNRegressionLoss(9)([deltas, bounding_box_targets, rpn_labels])
        scores = rpn_loss.RPNClassificationLoss(9)([scores, rpn_labels])

        proposals = object_proposal.ObjectProposal()([metadata, deltas, scores, anchors])

        proposals, labels_targets, bounding_box_targets = proposal_target.ProposalTarget()([proposals, labels, bounding_boxes])

        deltas, scores = _extract_regions(classes)([features, metadata, proposals])

        deltas = rcnn_loss.RCNNRegressionLoss()([deltas, bounding_box_targets, labels_targets])
        scores = rcnn_loss.RCNNClassificationLoss()([scores, labels_targets])

        bounding_boxes, scores = object_detection.ObjectDetection()([proposals, deltas, scores, metadata])

        outputs = [bounding_boxes, scores]

        super(RCNN, self).__init__(inputs, outputs)

    def compile(self, optimizer, **kwargs):
        super(RCNN, self).compile(optimizer, None)