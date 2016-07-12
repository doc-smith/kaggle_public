# I don't give a flying fuck about PEP8
class MatchTypes:
    EXACT        = 0
    BY_STEM      = 1
    BY_LEMMA     = 2
    BY_EXTENSION = 3


class Hit:
    def __init__(self, q_word, t_word, q_position, t_position, match_type):
        self.q_word = q_word
        self.t_word = t_word
        self.q_position = q_position
        self.t_position = t_position
        self.match_type = match_type

    def __repr__(self):
        return "{0} :: {1} -- {2} :: {3}, {4}".format(
            self.q_word,
            self.t_word,
            self.q_position,
            self.t_position,
            self.match_type
        )


def match_words(q, q_position, t, t_position):
    match = None
    if q.lowcase == t.lowcase:
        match = MatchTypes.EXACT
    elif q.morph_info.lemma == t.morph_info.lemma:
        match = MatchTypes.BY_LEMMA
    elif q.morph_info.stem == t.morph_info.stem:
        match = MatchTypes.BY_STEM

    if match is not None:
        return Hit(q, t, q_position, t_position, match)
    else:
        return None


def generate_hits(query, text):
    hits = []
    for t_position, t in enumerate(text):
        for q_position, q in enumerate(query):
            hit = match_words(q, q_position, t, t_position)
            if hit:
                hits.append(hit)
    return hits
