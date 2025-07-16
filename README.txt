MLSPred
1. Description
MLSPred is a missense mutation pathogenicity prediction method that uses the long-short distance attention mechanism (LSDA). MLSPred converts protein domain text features into feature vectors containing semantic information through the BioBERT model, concatenates them with other omics features into a feature matrix, and then uses the long-short distance attention mechanism to learn local information and global dependencies in the feature matrix, thereby improving the accuracy of missense mutation pathogenicity prediction.

2. Input data
MLSPred requires multi-omics signature data as input, and the demo input data is in the "data" folder.

3. Implementation
MLSPred is implemented in Python. It is tested on both Windows and Linux operating system. They are freely available for non-commercial use.

4. Usage
We provide two demo scripts to demonstrate how to run MLSPred.
4.1. To train the MLSPred model, run the following command in the command line:
python train.py
4.2. To test the effect of MLSPred on six independent test machines, run the following command in the command line:
python predict.py
This command will load the model trained on the training set and then make predictions on six independent test data.