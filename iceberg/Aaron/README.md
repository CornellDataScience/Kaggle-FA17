# Data Preprocessing and Augmentation

The Statoil/C-CORE Iceberg challenge provides competitors with ~1600 training sample and ~8200 testing samples of unprocessed Sentinel-1 data. 
Each sample includes the inclination angle, HH and HV polarized bands, and a value tag (iceberg or no iceberg)

Because we hope to build a convolutional neural network to classify the data, the number of training samples severly limits the depth of the models. 
Furthermore, the given data provides many corrupted data files. For many images, no incidence angle is provided. For others, backscatter renders the image almost unreadable.

My research goals aim at addressing these two problems as we construct deeper neural networks. 

## Data Preprocessing

The Sentinel-1 Satellites provides competitors with unprocessed SAR data. Due to the inherent properties of polarized data, the HH and HV bands don't provide us with enough information to cancel out most background noise. Furthermore, due to the inherent properties of SAR imaging, "speckles" of radar noise appear as random spikes.

#### Examples

Highly Speckled            | Less speckles            
:-------------------------:|:-------------------------:
![](images/Unfiltered_4.png)|![](images/Unfiltered_2.png)

### Filters

Filters provide us a way to remove speckle noise. Most of the filter code is adapted from the [pyradar](http://pyradar-tools.readthedocs.io/en/latest/) package. Example filters are given below

#### Filter Examples

Unfiltered          | Passed through Lee Filter          
:-------------------------:|:-------------------------:
![](images/Unfiltered_3.png)|![](images/Lee_Filtered_3.png)


