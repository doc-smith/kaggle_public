#!/usr/bin/env python

import collections
import json
import shelve
import sys

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import words


class Tokenizer(object):
    def __call__(self, words):
        def get_lemmas(words):
            return [word.morph_info.lemma for word in words]
        return get_lemmas(words)


def average_vector(mx):
    return mx.mean(axis=0)


def main():
    corpus = []

    print >>sys.stderr, "reading shelve.db..."
    storage = shelve.open("tmp/shelve.db")
    documents = dict(storage["documents"])
    print >>sys.stderr, "done"

    for doc in documents.itervalues():
        corpus.append(doc["query"])
        corpus.append(doc["title"])
        if doc["description"]:
            corpus.append(doc["description"])

    print >>sys.stderr, "reading amazon..."
    wg = words.WordGenerator()
    amazon_titles = {}
    with open("corpus/amazon.json", "r") as inp:
        amazon = json.loads(inp.read())
        for doc in amazon:
            amazon_words = wg.text_to_words(doc["title"])
            corpus.append(amazon_words)
            if doc["query"] not in amazon_titles:
                amazon_titles[doc["query"]] = [[]] * 10
            amazon_titles[doc["query"]][doc["rank"]] = amazon_words
    print >>sys.stderr, "done"

    print >>sys.stderr, "fit vectorizer..."
    vectorizer = TfidfVectorizer(
        tokenizer=Tokenizer(),
        min_df=3,
        encoding="unicode",
        max_features=None,
        ngram_range=(1, 2),
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        stop_words="english",
        lowercase=False
    )
    vectorizer.fit(corpus)
    print >>sys.stderr, "done"

    print >>sys.stderr, "computing amazon average vectors..."
    amazon_average_vectors = {}
    for query, ranked_title_words in amazon_titles.iteritems():
        amazon_average_vectors[query] = average_vector(
            vectorizer.transform(ranked_title_words)
        )

    print >>sys.stderr, "done"

    print ",".join((
        "doc_id",
        "amazon_avg_title_similarity",
        "amazon_max_similarity",
        "amazon_avg_similarity"))

    for doc_id, doc in documents.iteritems():
        avg_amazon_vector = amazon_average_vectors.get(doc["query_text"], None)
        if avg_amazon_vector is not None:

# FIXME: optimization
            title_vector = vectorizer.transform([doc["title"]])[0]
            description_vector = vectorizer.transform([doc["description"]])[0]
            avg_title_sim = cosine_similarity(title_vector, avg_amazon_vector)[0][0]

            sims = [0.0] * 10
            for rank, amazon_words in enumerate(amazon_titles[doc["query_text"]]):
                amazon_vector = vectorizer.transform([amazon_words])[0]
                sims[rank] = cosine_similarity(title_vector, amazon_vector)[0][0]
            max_sim = max(sims)
            avg_sim = np.average(sims)

            print ",".join(map(str, (doc_id, avg_title_sim,
                                     max_sim,
                                     avg_sim)))
        else:
            print "{},0.0,0.0,0.0".format(doc_id)

if __name__ == "__main__":
    main()
