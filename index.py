import joblib
import re
import string

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB

# read csv file with spaces and 
df = pd.read_csv('./data.csv', sep='\t', names=['message'])

# split messages when separator is found and save in an JSON array
messages = df.message.str.split('\t', expand=True).stack().reset_index(level=1, drop=True).rename('message')

# for every message separate in "label" and "text" when "^" is found
df = messages.str.split('^', expand=True).rename(columns={0:'TEXT', 1:'MESSAGE'})

print(df.head())



