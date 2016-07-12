#!/usr/bin/env python

import json
import requests
import sys
import traceback

import pandas as pd


API_URL = "https://api.datamarket.azure.com/Bing/Search/v1/SpellingSuggestions"
API_KEY = "HG0ETDzNghLyoMhxajP1LEOfSD5ieRDMbhEnRoRT+7s"


def main():
    csvs = sys.argv[1:]

    queries = set()

    for csv in csvs:
        data = pd.read_csv(csv, index_col=None)
        for query in data["query"]:
            queries.add(query)

    auth = requests.auth.HTTPBasicAuth(API_KEY, API_KEY)
    results = {
        "errors": [],
        "spelling_suggestions": {}
    }

    for query in queries:
        params = {
            "Query": "'{}'".format(query),
            "$format": "json"
        }

        print >>sys.stderr, query

        try:
            response = requests.get(API_URL, params=params, auth=auth)
            if response.status_code == 200:
                js = json.loads(response.text)

                suggestions = [ r["Value"] for r in js["d"]["results"] ]
                if suggestions:
                    results["spelling_suggestions"][query] = suggestions

                print >>sys.stderr, suggestions
                print >>sys.stderr, "ok"
            else:
                print >>sys.stderr, "status: {}".format(response.status_code)
                results["errors"].append(query)
        except Exception as err:
            print >>sys.stderr, "error: {}".format(err)
            traceback.print_exc(file=sys.stderr)
            results["errors"].append(query)

    print json.dumps(results, indent=4)


if __name__ == "__main__":
    sys.exit(main())
