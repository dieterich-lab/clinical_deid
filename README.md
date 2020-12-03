# Clinical DeID:<br> De-Identification of Clinical Admission Notes using BiLSTM and Embedded Languages Models (ELMo) as Word Representation

This repository includes the code of the publication [[Richter-Pechanski et al. 2019]](#1). This architecture was used to de-identify German medical admission notes from the cardiology domain in the HiGHmed Heidelberg partnerside. Main features are:

1. Pre-trained ELMo embeddings [[Wanxiang et al. 2018]](#3)[[Peters et al. 2018]](#4)
2. concatendated with Character Embeddings
3. Bidirectional LSTM layer

As this tool implemets a Keras model for named entity recognition, it can be performed on any sequence labeling task. Major parts of the code had been inspired by Tobias NER tutorial: https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/

Feel free to use the tool and adapt it to your needs. Usability improvements are very welcome.

## Prerequisities

* Python 3.5+
* Keras==2.3.1
* elmoformanylangs==0.0.2
* numpy==1.17.3
* seqeval==0.0.12
* scikit-learn==0.20.4

## Installation

This installation procedure has been tested on a Debian 8 machine. Attention: By default the tool generates ELMo embeddings with a maximum sequence length equal to the longest sequence in your data. In the i2b2 data this is 3039 token. To generate such embeddings, this code is tested on a machine with >1000G RAM.

1. Clone the repository
```console
foo@bar:~$ git clone https://github.com/dieterich-lab/clinical_deid.git
foo@bar:~$ cd clinical_deid
```
2. Create and activate a virtual environment and install required packages
```console
foo@bar:~$ virtualenv -p python 3.6 venv
foo@bar:~$ source venv/bin/activate
foo@bar:~$ pip install -r requirements.txt
```
3. Setup the ELMo embeddings using ELMoForManyLangs
    * Change into folder configs/elmo
    * Download the required language package from this repository: https://github.com/HIT-SCIR/ELMoForManyLangs
    * Unzip the zip file
    ```console
    foo@bar:~$ unzip 144.zip
    ```   
    * Change the value of the key config_path in the file config.json to cnn_50_100_512_4096_sample.json
    
Congratulations, now you are able to train a model for de-identification. 

## Data Format

If you want to run the i2b2 task, you need to download the i2b2 2006 de-identification data set at https://www.i2b2.org/NLP/DataSets/ and convert the data into a tab separated CoNLL file with 2 columns. The first column contains the Token, the second column the PHI class. Sentences or paragraphs are separated by a newline. 

```python
Mister   O
Thomas   PATIENT
Smith   PATIENT

68   AGE
Years   AGE
```
Currently the script does not use the BIO Format by default.

## Performaing de-identification task

1. To run the script, prepare your taining and test files as described above and save them into the folder *data*. Next edit the lstm_elmo.py script on line 17 and 18.
```python
# Defining the training and test data
path_train = 'data/deid_surrogate_train_all_version2.conll'
path_test = 'data/deid_surrogate_test_all_groundtruth_version2.conll'
```
2. For a custom evaluation report, you need to define your PHI class in the file lstm_elmo in line 14
3. Next you can run the script.
```console
foo@bar:~$ python lstm_elmo.py
```
3. If the training is done, you will get a tokenwise and entitywise classification report calculated on the test set on the console.
4. In addition a h5 model file called best_model_lstm_elmo.h5 is saved into the root folder. This can be used to load it into a de-identification pipeline.


## Folder structure

```bash
├── config
│   ├── elmo (Contains the pre-trained ELMo embeddings, see section Installation)
├── data (Containing CoNLL formated training and test files, see section Data Format)
```

## Evaluation on i2b2 2006 Data
Due to data protection reasons, we can not share the data set used in the publication [[Richter-Pechanski et al. 2019]](#1). <br>
For transparency and validity reasons, this model was evaluated on the dataset published for the Shared Task of the i2b2 challenge for De-Identification. For details see, [[Uzuner at al., 1007]](#2).
For evaluation the model had the following hyperparamters:<br>

### Hyperparameters
* Batch size: 128
* Epochs: 200
* Early stopping in *validation loss* with patience *20*
* Dimension ELMo: 1024
* Dimension Character Embedding: 15
* Maximum sequence length: longest sentence in dataset
* Validation split: 10% of training data

We did not do any hyperparamter tuning.

### Tokenwise Evaluation

|precision  |  recall|  f1-score |  support|
|--|--|--|--|
|       ID     |  0.99  |    0.99  |    0.99  |     755
|     DATE     |  0.99  |    1.00  |    0.99  |    1924
|   DOCTOR     |  0.96  |    0.96  |    0.96  |    1061
| LOCATION     |  0.81  |    0.71  |    0.75  |     119
| HOSPITAL     |  0.68  |    0.76  |    0.72  |     673
|  PATIENT     |  0.97  |    0.87  |    0.92  |     244
|    PHONE     |  0.91  |    0.88  |    0.89  |      58
|      AGE     |  0.00  |    0.00  |    0.00  |       3
|micro avg     |  0.93  |    0.94  |    0.93  |    4837
|macro avg     |  0.93  |    0.94  |    0.94  |    4837

### Entitywise Evaluation

|              precision |   recall  |f1-score  | support|
|--|--|--|--|
|         AGE|       0.00     | 0.00   |   0.00  |       3|
|        DATE|       0.99     | 1.00   |   0.99  |    2153|
|      DOCTOR|       0.98     | 0.98   |   0.98  |    2297|
|    HOSPITAL|       0.98     | 0.88   |   0.93  |    1598|
|          ID|       1.00     | 1.00   |   1.00  |    1194|
|    LOCATION|       0.96     | 0.79   |   0.87  |     240|
|     PATIENT|       0.99     | 0.94   |   0.96  |     510|
|       PHONE|       1.00     | 0.86   |   0.92  |      85|
|   micro avg|       0.99     | 0.96   |   0.97  |    8080|
|   macro avg|       0.86     | 0.80   |   0.83  |    8080|
|weighted avg|       0.99     | 0.96   |   0.97  |    8080|


## References
<a id="1">[1]</a> 
Richter-Pechanski P, Amr A, Katus HA, Dieterich C,
*Deep Learning Approaches Outperform Conventional Strategies in De-Identification of German Medical Reports.*, in
Stud Health Technol Inform. 2019 Sep 3;267:101-109,
doi:10.3233/SHTI190813.

<a id="2">[2]</a> 
Uzuner O, Luo Y, Szolovits P,
*Evaluating the state-of-the-art in automatic de-identification*, in
J Am Med Inform Assoc. 2007;14(5):550-563,
doi:10.1197/jamia.M2444.

<a id="3">[3]</a> 
Wanxiang Ch, Yijia L, Yuxuan W, Bo Zh, Ting L,
*Towards Better UD Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation*, in
Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies 2018, 55-64,
doi:10.18653/v1/K18-2005.

<a id="4">[4]</a>
Peters M,  Neumann M, Iyyer M, Gardner M, Clark Ch, Lee K, Zettlemoyer L,
*Deep contextualized word representations*
arXiv preprint arXiv:1802.05365 (2018).
