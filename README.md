>📋  A template README.md for code accompanying a Machine Learning paper

# Neural Fourier Space Modelling for Time Series

This repository is the official implementation of [Neural Fourier Space Modelling for Time Series](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Getting started
Clone the NFM repository via git as follows:

```clone
git clone ...
```

Then, install requirements as follows:

```setup
pip install -r requirements.txt
```
The current implementation uses a single GPU and provides no multiple GPU setup. 
<!-- There is a sub-folder for each task, under which you can find all task-specific codes, e.g., dataloaders, trainer, run scripts, etc.   -->

## Datasets
Download datasets from below links. 

1. You can download all forecastubg datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

2. For classification, running the experimental scripts (MFCC and raw) automaticall downloads the SpeechCommnad dataset and does necessary pre-process & formatting.

3. For anomaly detection, all the used datasets can be downloaded from [Anomaly Transformer](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm).

Create a seperate folder and save the downloaded datasets in the directory. Then, specify the location of datasets in the run scripts under each sub-folder.
We examplify ones in ETTm1 and speechcommand run scripts.

## Training & evaluation
We have separated run scrips and main ... for each task.  

Once you have downloaded the necessary datasets, you are all set to run the experiments.

### Forecasting
To train NFM on the conventional forecasting task (equal input and output resolution but different timespan), do
```trainf
sh ./Forecasting/scripts/ETTm.sh
```
- This will run NFM training on ETTm1 and ETTm2.
- The evaluation automatically follows after it.
- The .sh run script also contains "testing on different resolution outputs" which is also made after the evaluation on the conventional forecasting task. 
- Replace ETTm to others for training NFM on others.

### Classification
To train NFM for classification task on raw SpeechCommand, do 

```trainc
sh ./Classification/scripts/speechcommand_raw.sh 
```
This also runs both training and evaluations, including normal scenario and different sampling rate scenario.  


To train NFM on MFCC, do 
```trainc
sh ./Classification/scripts/speechcommand.sh 
```
This also runs evaluations afterwards.

### Anomaly Detection
To train NFM on anomaly detection task, do
```traina
sh ./AnomalyD/scripts/SMD.sh 
```
Replace SMD to others for training on others. 

Note that as mentioned in the main work, we found some flaws in the other's official implementation codes. 
We provide fixed code samples (in ... ) that we used to replace their original ones and to run the implementation codes. 

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## How to apply to your own dataset  
As presented in our main work, NFM can deal with different (explicitly accounting for the relation between inputs and outputs, which is a core principle to turn NFM into different learning modes, e.g., Forecasting, anomaly detection, classification, and so on) 

In the code, this is controlled by specifying m_t and m_f (during training or testing time) and accordingly applying to the model. 
This is done by a class object that has and manages globally (see "var.py"). We also provide a demo on how to set up NFM for your own dataset and use in notebook... .

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 