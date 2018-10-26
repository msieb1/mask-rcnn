### Changes made to model.py file to obtain ROI-Pooling features from MaskRCNN output module

* lines 2035 - 2059: Produce ROI-Pooling Features and function returns have been modified accordingly
			(only in the inference branch)

* line 962: yields roi_features now

* line 689 and 775: include roi_features now 

* line 818: added "a" to include roi_features as input

* lines 827 and 832: added shape of roi_feature maps

* lines 231 and 232: scaled up height and width to get larger ROI feature maps