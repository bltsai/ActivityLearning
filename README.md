# Activity Recognition
To run, you need to have and extract [data,activites,sensors package](https://drive.google.com/open?id=1hhvRKfI7zmJ0bRJvHhpITK-vTzIwQp3S)

and install packages through pip

```bash
pip3 install scikit-learn numpy lda
```

```bash
python3 ar_topic.py
```



# Activity Prediction
To run, you need to have [streaming_full.txt](https://drive.google.com/open?id=1EAmA0LjoyEGuJJ_mUbCF_FZR5Et8JAKU)

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
