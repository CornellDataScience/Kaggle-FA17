# Statoil/C-CORE Iceberg Classifier Challenge


<img src="https://storage.googleapis.com/kaggle-media/competitions/statoil/NM5Eg0Q.png" width="500">

### Overview
The goal of this Kaggle challenge is to build an algorithm that automatically identifies if a target found by remote sensing systems is a ship or iceberg. Participants need to analyze the shape, size and brightness of the object and its surrounding using data from two channels: HH (transmit/ receive horizontally) and HV (transmit horizontally/ receive vertically). Submissions will be evaluated on the log loss between the predicted values and the ground truth.

### Current Model

### Challenge
- There are only 1604 training examples and 8424 test examples, which makes the classification prone to overfitting.
- The dataset contains energy signals produced from radar backscatter, rather than ordinary RGB or black/white images.
- The "non-pixel" feature of each image, incidence angle, has "NA" missing data.

### Timeline
- 10/26 Research Proposal
- 10/30 Presentation

### Reference
- [Ship-Iceberg Discrimination with Convolutional Neural Networks in High Resolution SAR Images](http://elib.dlr.de/99079/2/2016_BENTES_Frost_Velotto_Tings_EUSAR_FP.pdf)

- [Very Deep Convolutional Neural Network Based Image Classification Using Small Training Sample Size](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7486599)

- [Multi-view Convolutional Neural Networks for 3D Shape Recognition](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf)

