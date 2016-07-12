#!/usr/bin/env python

import csv
import shelve
import sys

from collections import defaultdict

import numpy as np
import pandas as pd

import hits
import static_features
import words

from relevance import HitsTrackers


storage_filename = 'tmp/shelve.db'


def compute_avgdl(documents, text_name):
    lens = [ len(doc[text_name]) for doc in documents.itervalues() ]
    return np.average(lens)


def main():
    storage = shelve.open(storage_filename)

    sys.stderr.write('Reading from {0}...'.format(storage_filename))
    df = dict(storage['df'])
    documents = dict(storage['documents'])
    sys.stderr.write('done\n')

    avgdl_title = compute_avgdl(documents, "title")
    avgdl_description = compute_avgdl(documents, "description")

    number_of_documents = len(documents)
    trackers = (
        (HitsTrackers(df, number_of_documents, avgdl_title, "title"), "title_hits", "title"),
        (HitsTrackers(df, number_of_documents, avgdl_description, "description"), "description_hits", "description")
    )

    sf = static_features.StaticFeatures()

    columns = ['doc_id']
    for tracker, _, _ in trackers:
        columns.extend(tracker.names())
    columns.extend(sf.names())
    rows = []

    for doc_id in documents:
        document = documents[doc_id]
        features = []
        features.append(doc_id)
        for tracker, hits_name, text_name in trackers:
            tracker.new_doc(document['query'], document[text_name])
            for hit in document[hits_name]:
                tracker.add_hit(hit)
            features.extend(tracker.features())
        features.extend(sf.features(
            document["query"],
            document["title"],
            document["description"]
        ))
        rows.append(features)

    pd.DataFrame(rows, columns=columns).to_csv('tmp/features.csv', index=False)

    storage.close()

if __name__ == "__main__":
    sys.exit(main())
