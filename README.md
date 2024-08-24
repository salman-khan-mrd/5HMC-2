<h1><center>A Hybrid residue based sequential encoding mechanism with XGBoost Improved Ensemble Model for identifying 5-Hydroxymethylcytosine Modifications</h1></center>
To guarantee proper use of our code, please follow all the steps in the presented order.
<p> </p>
<h2>Introduction</h2>
XGBoost5hmC is an intelligent and robust machine learning model that adapts and enhances Chou's 5-step framework by leveraging the XGBoost algorithm to extract discriminative features for predicting 5-hydroxymethylcytosine (5hmC) modifications. Our improved model, XGB5hmC, advances the original machine learning approach by employing seven feature extraction techniques to convert RNA sequences into feature vectors, which are then combined into a fused feature vector. Additionally, SHAP analysis is used to select the most significant features, removing unnecessary and redundant information. Among several learning models tested, the XGBoost model trained with 10-fold cross-validation is chosen for its superior performance in training and prediction.
<h2>Python Packages and Dependencies</h2>
The model relies on the following Python packages and their specific versions:
<li>Pandas: 1.0.5</li>
<li>NumPy: 1.18.5
<li>Pip: 20.1.1
<li>SciPy: 1.4.1
<li>scikit-learn: 0.23.1
<li>TensorFlow: 2.3.1
<br>Please ensure you have a compatible Python environment with these dependencies to use the model effectively.
<h2>Create a Virtual Environment</h2>
To install the required dependencies for this project, you can create a virtual environment and use the provided requirements.txt file. Make sure your Python environment matches the specified versions above to ensure compatibility.

python -m venv myenv  # Create a virtual environment
source myenv/bin/activate  # Activate the virtual environment
pip install -r requirements.txt  # Install the project dependencies
<h1>Download Protein Sequences </h1>
In bioinformatics and machine learning, the acquisition or selection of a valid benchmark dataset is an essential step 
for developing an intelligent computational model. The selection of a suitable benchmark has a high impact on the performance of
a computational model. According to Chou's comprehensive review, 18,19 a valid and reliable benchmark dataset is always required for the design of a powerful 
and robust computational model. Hence, in this paper, we used the same benchmark datasets that were used in 16. 

We have included a file named datset.fasta as an example in this repository
Once you have downloaded the sequence you can send the sequence as input.
<h2>FEATURE EXTRACTION TECHNIQUES</h2>
n order to extract only the "K" sites from the contextualized embeddings you can use the following code:

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("P10275_Prot_Trans_.csv", header = None)                     # replace with your ProtT5 embeddings file
Header = ["Residue"]+[int(i) for i in range(1,1025)]
df.columns = Header
df_K_Residue_Only = df[df["Residue"].isin(["K"])]

K_residue_index = []
All_residue_list = list(df["Residue"])
for residue in range(len(All_residue_list)):
    index_no = residue+1
    amino_acid = All_residue_list[residue]
    if amino_acid == "K":
        K_residue_index.append(index_no)

df_K_Residue_Only["K_index_starting_from_one"] = K_residue_index

df_K_Residue_Only.to_csv("P10275_K_Sites.csv")                 # saves the embeddings of only K residues

df_K_Residue_Only = pd.read_csv("P10275_K_Sites.csv")

df_K_Residue_Only
Once the process is complete, you will have a .csv file containing the embeddings of "K" sites.
<h2>Machine Learning Algorithms:</h2>
<li>Support Vector Machine using Radial Basis Function Kernel
<li>Random Forests
<li>KNN
<li>XGBoost
<li>NB
<li>LR

<h2>How to Run the Program:</h2>
The usage of the program is as follows:
python main.py {raw_data} {data_annotations} {classifier 1-4} {down_sample - 0 or 1} {cross_validate - 0 or 1} {n_neighbors}

<h4>To select a classifier, supply a number 1-6 corresponding to:</h4>
<li>Support Vector Machine using Radial Basis Function Kernel
<li>Random Forests
<li>KNN
<li>XGBoost
<li>NB
<li>LR
To enable down sampling, supply 1. To disable, supply 0

<i>*** ONLY IF USING KNN:</i>

<h4>Supply a valid integer n_neighbors</h4>
Examples:

KNN with 5 neighbors, down sampling enabled, cross validation disabled:

- python main.py GSE60361C13005Expression.txt expressionmRNAAnnotations.txt 3 1 0 5

SVM with down sampling disabled, cross validation enabled

<i>- python main.py GSE60361C13005Expression.txt expressionmRNAAnnotations.txt 1 0 1</i>
<b>Description of Each File:</b>
main.py: This file processes the command line arguments supplied and runs the program using the selected classifier. Based on user input, the program decides whether to down sample, cross validate, and which classifier to use. This file defines the four classifiers supplied to the user for classification: Support Vector Machine using Radial Basis Function Kernel, Mutli-Layer Perceptron (Neural Network), K-Nearest Neighbor, and Random Forest. After calling the preproccess code and running the classification, this class sends the results to an analysis file that performs evaluations and writes the results to file.

xgboost.py This class object represents the RNA Seq Data. It holds the raw data, the annotations, and provides methods for partitioning the data. The partitions (for both down sampling and non down sampling and cross validation and no cross validation) randomly make partitions of the data for both training and testing, while simultaneously holding the annotations for the randomly selected testing data. The class also provides accessor methods for all data, annotations, training data, testing data, and training data target values to evaluate performance.

preprocess.py This file does all of the preprocessing work before running classification. This file loads raw data and annotations into memory. Next, this file is used to down dample by both cluster size and molecule count.

analysis.py After classification, this file is used to evaluate the performance of the classifier and write the results to an output file. First, this file creates a confusion matrix and then computes the accuracy, sensitivity, specificity, MCC, and F1 Score at both the class level and the global level across the supplied number of folds (10 folds for cross validation and 1 fold for non-cross validation). This class also uses a basic metric of merely counting the number of correct classifications that was used initially to check the performance of the classifiers. After evaluating, the results are written to a file.

knn.py This file defines the K Nearest Neighbor classifier. It allows the user to specificy the number of neighbors used and then fits the training data and the samples to the classifier. Then, it takes training data and makes predictions, returning the results of the predictions.

lr.py This file defines the Multi-Layer Perceptron (Neural Network). It fits the training data and the samples to the classifier. Then, it takes training data and makes predictions, returning the results of the predictions.

randomForest.py This file defines the Random Forest Classifier. It fits the training data and the samples to the classifier. Then, it takes training data and makes predictions, returning the results of the predictions.

NB.py This file defines the Support Vector Machine using a Radial Basis Function Kernel. 
It fits the training data and the samples to the classifier. Then, it takes training data and makes predictions, returning the results of the predictions.


<h2>Any Questions?</h2>
If you need any help don't hesitate to get in touch with Dr. Subash Chandra Pakhrin (salman-mrd@gmail.com)
