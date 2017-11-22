# Statoil/C-CORE Iceberg Classifier Challenge - Lily

### Overview
The goal of this Kaggle challenge is to build an algorithm that automatically identifies if a target found by remote sensing systems is a ship or iceberg. Participants need to analyze the shape, size and brightness of the object and its surrounding using data from two channels: HH (transmit/ receive horizontally) and HV (transmit horizontally/ receive vertically). Submissions will be evaluated on the log loss between the predicted values and the ground truth.

### Current Model
 A Convolutional Neural Network with 4-blocks. It handles multi-inputs: one meta data input (i.e. incidence angle) and one image input for image data with TWO channels: HH and HV. Four types of augmentations such as 'Flip', 'Rotate', 'Shift', 'Zoom' are applied before training. This ConvNet scores 0.2023.

### Challenge
- There are only 1604 training examples and 8424 test examples, which makes the classification prone to overfitting.
- The dataset contains energy signals produced from radar backscatter, rather than ordinary RGB or black/white images.
- The "non-pixel" feature of each image, incidence angle, has "NA" missing data.

### Research possibilities
- Multi-view CNN
- Region-based CNN

### Timeline
- 10/26 Research Proposal
- 10/30 Midterm Presentation
- 12/05 Final Presentation

### Reference
- [Contextual Region-Based Convolutional Neural Network with Multilayer Fusion for SAR Ship Detection](http://www.mdpi.com:8080/2072-4292/9/8/860)
- [Multi-view Convolutional Neural Networks for 3D Shape Recognition](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf)

