Using the datasets provided in IMDB and opinion-lexicon-English folders we train a binary classifier which is able to identify Positive and Negative movie reviews. The python code creates and tests different models (SVM) in an attempt to find the best classifier. 

For executing the code on multiple processors an extra parameter can be given.
e.g. "python3 sentiment_analysis.py 3" will execute on three cores. 
No input is sequential execution.  

Python version used:
- Python 3.6.9

Neccesary libraries:
- pandas
- numpy
- sklearn
- nltk
- re
- itertools
- joblib 
- multiprocessing
- time
- os
- sys

