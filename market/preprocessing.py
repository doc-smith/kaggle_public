#!/usr/bin/env python

import csv
import json
import shelve
import sys

from collections import defaultdict
import pandas as pd

import hits
import words


QUERY_SPELLING_FILENAME = "corpus/spelling.json"

def main():
    csvs = sys.argv[1:]
    documents = defaultdict(dict)
    df = defaultdict(int)
    max_df = {}

    with open(QUERY_SPELLING_FILENAME, "r") as inp:
        query_spelling = json.loads(inp.read())["spelling_suggestions"]

    for csv in csvs:
        data = pd.read_csv(csv, index_col=None)

        wg = words.WordGenerator()
        for index, row in data.iterrows():
            doc_id, query_text, title_text, desc_text \
                = row[["id", "query", "product_title", "product_description"]]
            title = wg.text_to_words(title_text.decode("utf-8"))
            documents[doc_id]["query_text"] = query_text

            if not pd.isnull(desc_text):
                cleaned = words.remove_html(desc_text.decode("utf-8"))
                description = wg.text_to_words(cleaned)
            else:
                description = []

            if query_text in query_spelling:
                fixed_query = query_spelling[query_text][0]
                query = wg.text_to_words(fixed_query.decode("utf-8"))
            else:
                query = wg.text_to_words(query_text.decode("utf-8"))

            doc = documents[doc_id]

            doc["title"] = title
            doc["description"] = description

            doc["title_hits"] = hits.generate_hits(query, title)
            doc["description_hits"] = hits.generate_hits(query, description)

            doc["query"] = query

            doc_df = defaultdict(int)
            for lemma in set((word.morph_info.lemma for word in title)):
                doc_df[lemma] += 1

            for lemma in set((word.morph_info.lemma for word in description)):
                doc_df[lemma] += 1

            max_df[doc_id] = max([v for v in doc_df.itervalues()])
            for lemma in doc_df:
                df[lemma] += 1

    storage = shelve.open("tmp/shelve.db", flag="n")
    storage["documents"] = documents
    storage["df"] = df
    storage["max_df"] = max_df
    storage.close()

if __name__ == "__main__":
    sys.exit(main())
