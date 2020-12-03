# Clinical DeID:<br> De-Identification of Clinical Admission Notes using BiLSTM and Embedded Languages Models (ELMo) as Word Representation

This repository includes the code of the publication [[Richter-Pechanski et al. 2019]](#1)

## Evaluation on i2b2 2006 Data
This model was evaluated on the dataset published for the Shared Task of the i2b2 challenge for De-Identification. for details see, [[Uzuner at al., 1007]](#2).
For evaluation the model had the following hyperparamters:<br>

### Hyperparameters
* Batch size: 128
* Epochs: 200
* Early stopping in *validation loss* with patience *20*
* Dimension ELMo: 1024
* Dimension Character Embedding: 15
* Maximum sequence length: 3037
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
Richter-Pechanski P, Amr A, Katus HA, Dieterich C.,
*Deep Learning Approaches Outperform Conventional Strategies in De-Identification of German Medical Reports.*, in
Stud Health Technol Inform. 2019 Sep 3;267:101-109,
doi: 10.3233/SHTI190813. PMID: 31483261.

<a id="2">[2]</a> 
Uzuner O, Luo Y, Szolovits P.,
*Evaluating the state-of-the-art in automatic de-identification*, in
J Am Med Inform Assoc. 2007;14(5):550-563,
doi:10.1197/jamia.M2444.

