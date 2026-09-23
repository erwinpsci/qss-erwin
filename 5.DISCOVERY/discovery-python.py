# discovery-python.py
# Python equivalent of the initial R setup in Chapter 5 (Discovery)

# --- core libraries ---

# %% [code]
# standard libraries ---
import os
import re
import pandas as pd
import numpy as np

# %% [code]
# --- text processing ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# --- machine learning / matrices ---
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- visualization (used later) ---
import matplotlib.pyplot as plt

# --- ensure required NLTK resources are available ---
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

# --- initialize objects analogous to R setup ---
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# %% [code]
# --- directory holding Federalist text files (fp01.txt, fp02.txt, ...) ---
DIR_SOURCE = "/Users/haoxi-home/Desktop/因果推断/qss-erwin/DISCOVERY/federalist"   # adjust path as needed

# --- load raw corpus (equivalent to VCorpus + DirSource) ---
file_names = sorted(
    f for f in os.listdir(DIR_SOURCE) if f.startswith("fp") and f.endswith(".txt")
)

corpus_raw = []
doc_ids = []

for fname in file_names:
    with open(os.path.join(DIR_SOURCE, fname), "r", encoding="utf-8") as f:
        corpus_raw.append(f.read())
        doc_ids.append(fname)

# --- tidy-style corpus (equivalent to tidy(corpus_raw) %>% select(id, text)) ---
corpus_df = pd.DataFrame({
    "id": doc_ids,
    "text": corpus_raw
})

# --- extract numeric document id (fp01.txt -> 1) ---
corpus_df["new_id"] = corpus_df["id"].str.extract(r"(\d+)").astype(int)

# preview
corpus_df.head()

# %% [code]
# --- tokenize, stem, clean, and filter 

from nltk.tokenize import word_tokenize


# %%
rows = [] # list to hold processed tokens

for _, row in corpus_df.iterrows():
    doc_id = row["id"]
    new_id = row["new_id"]
    text = row["text"].lower()  # lowercase

    # tokenize into words
    tokens = word_tokenize(text)

    for word in tokens: # tokens is a list of words, which comes from word_tokenize
        # remove numbers 
        word_clean = re.sub(r'\d+', '', word) # `sub` is a function to replace patterns
        # drop empty strings
        if word_clean == "": #`if` is a conditional statement, which means if the condition is met,
            continue # skip to next iteration 
        # stem the word
        stem = stemmer.stem(word_clean) # stem is the stemmed version of word_clean. It is a function that reduces words to their root form.

        rows.append({ # append a dictionary to the list, which is a function to add an element to a list
            "id": doc_id,
            "new_id": new_id,
            "word": word_clean,
            "stem": stem
        })

tokens_raw = pd.DataFrame(rows)
# preview
tokens_raw.info()

# remove stop words
# %%
tokens = tokens_raw[~tokens_raw["word"].isin(stop_words)].copy()
# equivalent to glimpse(tokens)
tokens.info()


# %% [code]
# extract essay no. 10 (R: corpus_raw[[10]])
essay_10 = corpus_raw[9]   # Python is 0-indexed

# print full text (R: content())
print(essay_10)

# check stpo words
tokens[tokens["word"].isin(["the", "and", "of", "to", "is"])].head()

tokens["word"].value_counts().head(20)

tokens_raw["word"].value_counts().head(10)
tokens["word"].value_counts().head(10)