# Modified implementation codes

For baseline results we have in our work, make the following correction in their official implementation codes

## Anomaly detection

Replace the dataloader classes for anomaly detection datasets (PSM, SMAP, SMD, and MSL) in [Anomaly Transformer](https://drive.google.com/drive/folders/), [TimesNet](https://github.com/thuml/TimesNet), and [FITS](https://anonymous.4open.science/r/FITS), with the following [code](etc/modified_ad_dataload.py).

The following changes are made 
- Set correct validation dataset (split from training set)
- Apply correct standardization to validation and testing set


Moreover, replace the test function in anomaly dectection solver script of the above implementations with the following [code](etc/modified_ad_eval.py)

This makes the following changes.
- Compute threshold using correct validation set and training set.
- Evaluate on correct testing set.   


## Forecasting

For those who are willing to conduct the experiment of forecasting at different sampling rates in models like [PatchTST](https://github.com/yuqinie98/PatchTST) and [N-linear](https://github.com/cure-lab/LTSF-Linear), 

Add the following [lines of code](modified_fore_dataload.py) to the dataloader of their original implementation.  

This applies downsampling followed by resampling through zero-padding in frequency domain (equivalent to applyng sinc kernel interpolation) to testing data only. 