def query_has_digit(query):
    for word in query:
        if word.lowcase.isdigit():
            return 1.0
    return 0.0


class StaticFeatures:
    def names(self):
        return [
            "query_length",
            "title_length",
            "description_length",
            "query_has_digit"
        ]

    def features(self, query, title, description):
        return [
            len(query),
            len(title),
            len(description),
            query_has_digit(query)
        ]
