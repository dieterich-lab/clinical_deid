# Clinical DeID:<br> De-Identification of Clinical Admission Notes using BiLSTM and Embedded Languages Models (ELMo) as Word Representation

This repository includes the code of the publication [[Richter-Pechanski et al. 2019]](#1). This architecture was used to de-identify German medical admission notes from the cardiology domain in the HiGHmed Heidelberg partnersite. Main features are:

1. Pre-trained ELMo embeddings [[Wanxiang et al. 2018]](#3)[[Peters et al. 2018]](#4)
2. concatendated with Character Embeddings
3. Bidirectional LSTM layer

As this tool implements a Keras model for named entity recognition, it can be applied on any sequence labeling task. Parts of the code are inspired by Tobias' amazing NER tutorial: https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/

Feel free to adapt the tool to your needs. Functionality and usability improvements are very welcome.

## Prerequisities

* Python 3.5+
* Keras==2.3.1
* elmoformanylangs==0.0.2
* numpy==1.17.3
* seqeval==0.0.12
* scikit-learn==0.20.4

Pre-trained ELMo embeddings. Where to get and how to setup see next section (Installation) bullet 3.

## Installation

This installation procedure has been tested on a Debian 8 machine. Attention: By default the tool generates ELMo embeddings with a maximum sequence length equal to the longest sequence in your data. In the i2b2 data this is 3039 token. To generate such embeddings, this code is tested on a machine with >1000G RAM.

1. Clone the repository
```console
foo@bar:~$ git clone https://github.com/dieterich-lab/clinical_deid.git
foo@bar:~$ cd clinical_deid
```
2. Create and activate a virtual environment and install required packages
```console
foo@bar:~$ virtualenv -p python3.6 venv
foo@bar:~$ source venv/bin/activate
foo@bar:~$ pip install -r requirements.txt
```
3. Setup the ELMo embeddings using ELMoForManyLangs
    * Change into folder `configs/elmo/` 
    ```console
    foo@bar:~$ cd configs/elmo/
    ```
    * Download the required language package from this repository: https://github.com/HIT-SCIR/ELMoForManyLangs
    * Unzip the zip file
    ```console
    foo@bar:~$ unzip 144.zip
    ```   
    * Change the value of the key *config_path* in the file `config.json` to `cnn_50_100_512_4096_sample.json`.
    
Congratulations, now you are able to train a model for de-identification. 

## Data Format

If you want to run the i2b2 task, you need to download the i2b2 2006 de-identification data set at https://www.i2b2.org/NLP/DataSets/ and convert the data into a tab separated CoNLL file with 2 columns. The first column contains the token, the second column the entity class. Sentences or paragraphs are separated by a newline. 

```python
Mister   O
Thomas   PATIENT
Smith   PATIENT

08   DATE
2010   DATE
```

## Performing a de-identification task

1. To run the script, prepare your CoNLL formated training and test files as described in section *Data Format* and save them into the folder `data/`.
2. Next you can run the script defining the arguments `--path_train` and `--path_test` and `--mode [binary|multiclass]`. E.g. performing a binary PHI training:
```console
foo@bar:~$ python lstm_elmo.py --mode binary --path_train data/deid_surrogate_train_all_version2.conll --path_test data/deid_surrogate_test_all_groundtruth_version2.conll
```
3. If the training is done, you will get a tokenwise and entitywise classification report calculated on the test set printed to STDOUT.
4. In addition a h5 model file called `best_model_lstm_elmo.h5` is saved into the folder `models/`. This can be used to load it into a de-identification pipeline.

## Customizing hyperparameters

Currently, to customize the hyperparamters, you have to edit them in the script `lstm_elmo.py`.
* Batch_size is defined in line 164
* Number of epochs in line 165
* Hidden layer sizes, drop out values, and the loss function are defined in lines 179-201
* Early stopping and patience are defined in line 204-205


## Folder structure

```bash
├── config
│   ├── elmo (Pre-trained ELMo embeddings, see section Installation)
├── data (CoNLL formated training and test files, see section *Data Format*)
├── models (H5 formated models saved by recent training sessions) 
├── embeddings (Generated ELMo embeddings of training and test data) 
```

## Evaluation on i2b2 2006 Data
Due to data protection reasons, we can not share the data set used in the publication [[Richter-Pechanski et al. 2019]](#1). <br>
For transparency and validity reasons, this model was evaluated on the dataset published for the Shared Task of the i2b2 challenge for De-Identification. For details see, [[Uzuner at al., 2007]](#2).
For evaluation the model had the following hyperparamters:<br>

### Hyperparameters, used for the i2b2 2006 Evaluation
* Batch size: 128
* Epochs: 100
* Early stopping in *validation loss* with patience *10*
* Dimension ELMo: 1024
* Dimension Character Embedding: 15
* Maximum sequence length: longest sentence in dataset
* Validation split: 10% of training data
* Hidden layer size LSTM: 50 (2x BiLSTM: 100)
* Dropout after embedding layers: 0.3

We did not do any further hyperparamter tuning.

### Entitywise Evaluation
Multiclass

||precision  |  recall|  f1-score |  support|
|--|--|--|--|--|
|     DATE    |   0.99|      0.99|      0.99|      1924|
|   DOCTOR    |   0.96|      0.95|      0.96|      1061|
| HOSPITAL    |   0.68|      0.77|      0.72|       673|
|       ID    |   0.99|      0.99|      0.99|       755|
|  PATIENT   |    0.97|      0.92|      0.94|       244|
|    PHONE  |     0.90|      0.91|      0.91|        58|
| LOCATION |      0.82|      0.71|      0.76|       119|
|micro avg|       0.93|      0.94|      0.94|      4837|
|macro avg|       0.93|      0.94|      0.94|      4837|

Binary 


||precision  |  recall|  f1-score |  support|
|--|--|--|--|--|
|      PHI  |     0.93     | 0.94|      0.93|      4104|

### Tokenwise Evaluation
Multiclass

||precision  |  recall|  f1-score |  support|
|--|--|--|--|--|
|        DATE|       1.00|      0.99|      0.99 |     2153|
|          ID|       0.99|      1.00|      1.00 |     1194|
|    LOCATION|       0.98|      0.82|      0.89 |      240|
|       PHONE|       1.00|      0.91|      0.95 |       85|
|     PATIENT|       0.99|      0.96|      0.97 |      510|
|    HOSPITAL|       0.99|      0.88|      0.93 |     1598|
|      DOCTOR|       0.99|      0.98|      0.98 |     2297|
|   micro avg|       0.99|      0.96|      0.97 |     8080|
|   macro avg|       0.87|      0.82|      0.84 |     8080|
|weighted avg|       0.99|      0.96|      0.97 |     8080|

Binary 

||precision  |  recall|  f1-score |  support|
|--|--|--|--|--|
|         PHI   |    0.99  |    0.96   |   0.98  |    8080|

## References
<a id="1">[1]</a> 
Richter-Pechanski P, Amr A, Katus HA, Dieterich C,
*Deep Learning Approaches Outperform Conventional Strategies in De-Identification of German Medical Reports.*, in
Stud Health Technol Inform. 2019 Sep 3;267:101-109,<br/> 
doi:10.3233/SHTI190813.

<a id="2">[2]</a> 
Uzuner O, Luo Y, Szolovits P,
*Evaluating the state-of-the-art in automatic de-identification*, in
J Am Med Inform Assoc. 2007;14(5):550-563,<br/> 
doi:10.1197/jamia.M2444.

<a id="3">[3]</a> 
Wanxiang Ch, Yijia L, Yuxuan W, Bo Zh, Ting L,
*Towards Better UD Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation*, in
Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies 2018, 55-64,<br/> 
doi:10.18653/v1/K18-2005.

<a id="4">[4]</a>
Peters M,  Neumann M, Iyyer M, Gardner M, Clark Ch, Lee K, Zettlemoyer L,
*Deep contextualized word representations*
arXiv preprint arXiv:1802.05365 (2018).
