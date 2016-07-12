import json
import collections

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sys import exit
from util import save_csr

import words


# array declarations
s_data = []
s_labels = []
t_data = []
t_labels = []

#load data
train = pd.read_csv("data/train.csv").fillna("")
test  = pd.read_csv("data/test.csv").fillna("")


Document = collections.namedtuple("Document", ["query", "title", "description"])


QUERY_SPELLING_FILENAME = "corpus/spelling.json"
with open(QUERY_SPELLING_FILENAME, "r") as inp:
    query_spelling = json.loads(inp.read())["spelling_suggestions"]


def process_query(query):
    return query_spelling.get(query, [query])[0].decode("utf-8")


for i in range(len(train.id)):
    s_data.append(Document(
        process_query(train["query"][i]),
        train["product_title"][i].decode("utf-8"),
        train["product_description"][i].decode("utf-8")
    ))
    s_labels.append(str(train["median_relevance"][i]))

for i in range(len(test.id)):
    t_data.append(Document(
        process_query(test["query"][i]),
        test["product_title"][i].decode("utf-8"),
        test["product_description"][i].decode("utf-8")
    ))


class Tokenizer(object):
    def __init__(self, fields):
        self.generator = words.WordGenerator()
        self.fields = fields

    def __call__(self, doc):
        def get_lemmas(text):
            cleaned = words.remove_html(text)
            return [word.morph_info.lemma for word in self.generator.text_to_words(cleaned) ]

        tokens = []
        for field in self.fields(doc):
            tokens.extend(get_lemmas(field))

        return tokens

def tf_idf(train, test, min_df, fields, prefix):
    vectorizer = TfidfVectorizer(
        tokenizer=Tokenizer(fields),
        min_df=min_df,
        max_df=500,
        encoding="unicode",
        max_features=None,
        ngram_range=(1, 2),
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        stop_words="english",
        lowercase=False
    )

    print("[!] Processing TF-IDF for {0}".format(prefix))
    tf_idf_train = vectorizer.fit_transform(train)
    tf_idf_test = vectorizer.transform(test)
    save_csr('tmp/{0}_tf_idf_train'.format(prefix), tf_idf_train)
    save_csr('tmp/{0}_tf_idf_test'.format(prefix), tf_idf_test)

tf_idf(s_data, t_data, 1, lambda doc: [doc.query], 'queries')
tf_idf(s_data, t_data, 1, lambda doc: [doc.title], 'titles')
tf_idf(s_data, t_data, 5, lambda doc: [doc.description], 'description')
tf_idf(s_data, t_data, 5, lambda doc: [doc.title, doc.description], 'title_description')

