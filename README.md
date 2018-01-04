# Activity Recognition
Before you run, you need to have and extract [data,activites,sensors package](https://drive.google.com/open?id=1hhvRKfI7zmJ0bRJvHhpITK-vTzIwQp3S)

and install packages through pip

```bash
pip3 install scikit-learn numpy lda
```


```bash
sh ar_topic.sh
```

In the bottom of the code, choose to run either of the task function.

```python
task_original()
```
This is the original Prof. Yan's task: train-test data (75-25 split based on event total numbers) and trained the topic model, and mutual information matrix using the complete stream data.

```python
task_baseline()
```
This is the modified task: train-test data (75-25 split based on date) and trained the topic model, and mutual information matrix using only those 75% stream data. It doesn't filter the testing data.


```pythont
task_testing_blacklist()
```
This is the modified task with blacklist: train-test data (75-25 split based on date) and trained the topic model, and mutual information matrix using only those 75% stream data. It should filter the testing data. [TODO: implement the isBlacklist function to filter the sensor events]


# Activity Prediction
Before you run, you need to have [streaming_full.txt](https://drive.google.com/open?id=1EAmA0LjoyEGuJJ_mUbCF_FZR5Et8JAKU)

and install packages through pip

```bash
pip install matplotlib scikit-learn numpy seqlearn
```

```bash
python hmm_preprocess.py [0-31]
python hmm_ar.py
```

Or, to go through all feature combinations

```bash
sh hmm_for.sh
```
