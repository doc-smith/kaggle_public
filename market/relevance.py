import numpy as np

from hits import MatchTypes
from math import log


class StrikeCounter:

    def new_doc(self, query, text):
        self.length = len(query)
        self.previous_hit = None
        self.strike = None
        self.max_strike = 0

    def add_hit(self, hit):
        if hit.match_type != MatchTypes.EXACT:
            return

        if self.previous_hit is None:
            self.previous_hit = hit
            self.strike = 1
            self.max_strike = 1
            return

        if hit.q_position - self.previous_hit.q_position == 1 and \
           hit.t_position - self.previous_hit.t_position == 1 and \
           hit.t_word.context.sentence_index == self.previous_hit.t_word.context.sentence_index:
            self.strike += 1
            self.max_strike = max(self.strike, self.max_strike)
        else:
            self.strike = 1

        self.previous_hit = hit

    def get_max_strike(self):
        return self.max_strike


    def get_query_len(self):
        return self.length


class PhraseMatch(StrikeCounter):

    def __init__(self):
        self.name = 'phrase_match'

    def calculate(self):
        return 1 if self.get_max_strike() == self.get_query_len() else 0


class Proximity(StrikeCounter):

    def __init__(self):
        self.name = 'proximity'

    def calculate(self):
        return self.get_max_strike() / float(self.get_query_len())


class ExactMatch:

    def __init__(self):
        self.name = 'exact_match'

    def new_doc(self, query, text):
        self.matches = [False for _ in enumerate(query)]

    def add_hit(self, hit):
        if hit.match_type == MatchTypes.EXACT:
            self.matches[hit.q_position] = True

    def calculate(self):
        return 1 if all(self.matches) else 0


class MatchedProportion:

    def __init__(self):
        self.name = 'matched_proportion'

    def new_doc(self, query, text):
        self.matches = [0 for _ in enumerate(query)]

    def add_hit(self, hit):
        self.matches[hit.q_position] = 1

    def calculate(self):
        return np.average(self.matches)


class TfIdf:

    def __init__(self, df, number_of_documents):
        self.df = df
        self.name = 'tf_idf'
        self.number_of_documents = float(number_of_documents)

    def new_doc(self, query, text):
        self.query = query
        self.scores = [.0 for i, _ in enumerate(query)]

    def add_hit(self, hit):
        self.scores[hit.q_position] += 1

    def calculate(self):
        result = .0
        for i, e in enumerate(self.query):
            if e.morph_info.lemma in self.df:
                tf = self.scores[i]
                result += log(1.0 + tf) * log(self.number_of_documents / self.df[e.morph_info.lemma])
        result /= len(self.query)
        return result


class FirstWord:

    def __init__(self):
        self.name = 'first_word'

    def new_doc(self, query, text):
        self.match = False

    def add_hit(self, hit):
        if hit.t_position == 0 and hit.t_word.context.sentence_index == 0:
            self.match = True

    def calculate(self):
        return 1 if self.match else 0


class FirstSentenceMatchCount:

    def __init__(self):
        self.name = 'first_sentence_match_count'

    def new_doc(self, query, text):
        self.match_count = 0
        self.query_len = float(len(query))

    def add_hit(self, hit):
        if hit.t_word.context.sentence_index == 0:
            self.match_count += 1

    def calculate(self):
        return self.match_count / self.query_len


class BM25:

    def __init__(self, df, number_of_documents, avgdl, b, k1):
        self.df = df
        self.number_of_documents = number_of_documents
        self.avgdl = avgdl
        self.b = b
        self.k1 = k1
        self.name = "bm25_{}_{}".format(b, k1).replace(".", "_")

    def new_doc(self, query, text):
        self.doc_len = len(text)
        self.query = query
        self.scores = [0.0 for i, _ in enumerate(query)]

    def add_hit(self, hit):
        self.scores[hit.q_position] += 1

    def calculate(self):
        k = self.k1 * (1 - self.b + (self.b * self.doc_len) / self.avgdl)
        result = 0.0
        for i, e in enumerate(self.query):
            if e.morph_info.lemma in self.df:
                f = self.scores[i]
                idf = log(float(self.number_of_documents) / self.df[e.morph_info.lemma])
                result += idf * ((f * (self.k1 + 1)) / (f + k))
        return result / len(self.query)


class AdvancedProximity:

    def __init__(self, df, number_of_documents, avgdl, b, k1):
        self.df = df
        self.number_of_documents = number_of_documents
        self.avgdl = avgdl
        self.b = b
        self.k1 = k1
        self.name = "advanced_proximity_{}_{}".format(b, k1).replace(".", "_")

    def new_doc(self, query, text):
        self.doc_len = len(text)
        self.query = query
        self.acc = [0.0 for i, _ in enumerate(query)]
        self.previous_hit = None

    def add_hit(self, hit):
        if self.previous_hit:
            q_word = hit.q_word
            p_q_word = self.previous_hit.q_word
            if (p_q_word.morph_info.lemma != q_word.morph_info.lemma and
                p_q_word.morph_info.lemma in self.df and
                q_word.morph_info.lemma in self.df):

                idf = log(float(self.number_of_documents) / self.df[q_word.morph_info.lemma])
                prev_idf = log(float(self.number_of_documents) / self.df[p_q_word.morph_info.lemma])
                dscore = (hit.t_position - self.previous_hit.t_position) ** 2

                self.acc[hit.q_position] += prev_idf / dscore
                self.acc[self.previous_hit.q_position] += idf / dscore

        self.previous_hit = hit

    def calculate(self):
        k = self.k1 * (1 - self.b + (self.b * self.doc_len) / self.avgdl)
        result = 0.0
        for i, e in enumerate(self.query):
            if e.morph_info.lemma in self.df:
                idf = log(float(self.number_of_documents) / self.df[e.morph_info.lemma])
                result += idf * ((self.acc[i] * (self.k1 + 1)) / (self.acc[i] + k))
        return result / len(self.query)


class BCLM:

    def __init__(self, df, number_of_documents, avgdl, b, k1):
        self.bm25 = BM25(df, number_of_documents, avgdl, b, k1)
        self.proximity = AdvancedProximity(df, number_of_documents, avgdl, b, k1)
        self.name = "bclm_{}_{}".format(b, k1).replace(".", "_")

    def new_doc(self, query, text):
        self.query = query
        self.bm25.new_doc(query, text)
        self.proximity.new_doc(query, text)

    def add_hit(self, hit):
        self.bm25.add_hit(hit)
        self.proximity.add_hit(hit)

    def calculate(self):
        return self.bm25.calculate() + self.proximity.calculate()



class HitsTrackers:

    def __init__(self, df, number_of_documents, avgdl, group_name):
        self.group_name = group_name
        self.trackers = [
                          TfIdf(df, number_of_documents),
                          ExactMatch(),
                          PhraseMatch(),
                          Proximity(),
                          MatchedProportion(),
                          FirstWord(),
                          FirstSentenceMatchCount(),
                          BM25(df, number_of_documents, avgdl, b=0.5, k1=1.2),
                          BM25(df, number_of_documents, avgdl, b=0.0, k1=1.2),
                          AdvancedProximity(df, number_of_documents, avgdl, b=0.5, k1=1.2),
                          BCLM(df, number_of_documents, avgdl, b=0.5, k1=1.2)
                        ]

    def new_doc(self, query, text):
        for tracker in self.trackers:
            tracker.new_doc(query, text)

    def add_hit(self, hit):
        for tracker in self.trackers:
            tracker.add_hit(hit)

    def features(self):
        return [tracker.calculate() for tracker in self.trackers]

    def names(self):
        return [self.group_name + "_" + tracker.name for tracker in self.trackers]
