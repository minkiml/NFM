# Modified implementation codes

For some baseline results we have in our work, we found some bugs in their original implementation codes and make the following correction in them to produce the results presented in our work.

## Anomaly detection

Replace the dataloader classes for anomaly detection datasets (PSM, SMAP, SMD, and MSL) in [Anomaly Transformer](https://drive.google.com/drive/folders/), [TimesNet](https://github.com/thuml/TimesNet), and [FITS](https://anonymous.4open.science/r/FITS), with the following [code](modified_ad_dataload.py).

The following changes are made. 
- Set correct validation dataset (split from training set).
- Apply correct standardization to validation and testing set.


Moreover, replace the test function in anomaly dectection solver script of the above implementations with the following [code](modified_ad_eval.py).

This makes the following changes.
- Compute anomaly threshold using correct validation set and training set.
- Evaluate on correct testing set.   
- Compute correct anomaly threshold and do "point-wise" estimation (the process before the anomaly evaluation) in FITS where the estimation was used to be made over the entire window chunk.  


## Forecasting

To the best of our knowledge, the task of forecasting at different sampling rates is done for the first time in our work. For those who are willing to conduct the experiment and reproduce the results using models like [PatchTST](https://github.com/yuqinie98/PatchTST) and [N-linear](https://github.com/cure-lab/LTSF-Linear), add the following [lines of code](modified_fore_dataload.py) to the dataloader of their original implementation.  

This applies downsampling followed by resampling through zero-padding in frequency domain (equivalent to applyng sinc kernel interpolation) to testing data only. 

For FITS, we do not resample the downsampled inputs during data preparation. 
Then, if the frequency of the downsampled inputs is shorter than the training cut-off frequency during inference time, we zero-pad the shorter input frequency up to the training cut-off frequency.