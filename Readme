# Natural Language Processing Systems for Pathology Parsing in Limited Data Environments with Uncertainty Estimation

## Objective
Cancer is a leading cause of death, but much of the diagnostic information is stored as unstructured data in pathology reports. We aim to improve uncertainty estimates of machine-learning based pathology parsers and evaluate performance in low data settings.

## Materials and Methods
Our data comes from the Urologic Outcomes Database at UCSF which includes 3,232 annotated prostate cancer pathology reports from 2001-2018. We approach 17 separate information extraction tasks, involving a wide range of pathologic features. To handle the diverse range of fields we required two statistical models, a document classification method for pathologic features with a small set of possible values and a token extraction method for pathologic features with a large set of values. For each model, we used isotonic calibration to improve the model’s estimates of its likelihood of being correct. 

## Results
Our best document classifier method, a convolutional neural network, achieves a weighted F1 score of 0.97 averaged over 12 fields and our best extraction method achieves an accuracy of 0.93 averaged over 5 fields. The performance saturates as a function of dataset size with as few as 128 data points. Furthermore, while our document classifier methods have reliable uncertainty estimates, our extraction based methods do not, but after isotonic calibration, expected calibration error drops to below 0.03 for all extraction fields. 

## Conclusions
We find that when applying machine learning to pathology parsing, large datasets may not always be needed, and that calibration methods can improve the reliability of uncertainty estimates.


# Technical Details
This repository contains the codebase for extracting data from prostate reports. There are two high-level approaches. The token extraction approach, which should be used to extract continuous data fields or categorical data fields with a large possible set of values is exemplified in main_pipelines/token_extraction/RandomForest.ipynb notebook. The classification approach, which should be used for categorical data with a small number of possible values is referenced in main_pipelines/classification/ConvolutionalNetwork.ipynb and main_pipelines/classification/LogisticRegressionWithCalibration.ipynb. The latter notebook also contains code for calibrating probabilities. For small training sizes (in the hundreds or less), the non-deep learning approaches should be used.

Because the data used to train these models are protected, the data and the trained models cannot be public. The data structure were preprocessed from the raw data and are Python dictionaries. The 'train', 'val', and 'test' keys denote the split of the data. The corresponding values are Python lists of dictionaries representing each patient in the data. The labels and text of each patient are accessed through the patient dictionaries. 

For example, data['train'][i]['document'] contains the pathology report as a string for the ith patient in the training split, while data['train'][i]['labels']['TumorType'] contains the label for the data field "Tumor Type" for the ith pateint in the training split. 

