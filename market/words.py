import collections
import nltk

from bs4 import BeautifulSoup


WordContext = collections.namedtuple("WordContext", ["word_index",
                                                     "sentence_index"])


class MorhInfo:
    def __init__(self, stem, lemma):
        self.stem = stem
        self.lemma = lemma

    def __repr__(self):
        return u"{0}::{1}".format(self.lemma, self.stem)


class Word:
    def __init__(self, text, lowcase,
                 morph_info,
                 context):
        self.text = text
        self.lowcase = lowcase
        self.morph_info = morph_info
        self.context = context

    def __repr__(self):
        return u"{0} ({1}::{2}, {3}:{4})".format(
            self.text,
            self.morph_info.lemma,
            self.morph_info.stem,
            self.context.sentence_index,
            self.context.word_index).encode("utf-8")


def is_digit(word):
    return word.text.isdigit()


class MorphProcessor:
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.lemmatizer = nltk.WordNetLemmatizer()

    def process(self, text):
        return MorhInfo(
            self.stemmer.stem(text),
            self.lemmatizer.lemmatize(text)
        )


# FIXME: remove css
def remove_html(text):
    bs = BeautifulSoup(text)
    return bs.get_text(separator=" ")


EX_STOPWORDS = set([
    "http",
    "www",
    "img",
    "border"
])


class Tokenizer:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words("english")

    def tokenize(self, text):
        sentence_index = 0
        for line in text.split("\n"):
            sentences = nltk.sent_tokenize(line)
            for sentence in sentences:
                words = nltk.wordpunct_tokenize(sentence)
                word_index = 0
                for word_text in words:
                    if self.__check_word(word_text):
                        yield word_text, WordContext(word_index, sentence_index)
                        word_index += 1
                sentence_index += 1

    def __check_word(self, text):
        lowcase = text.lower()
        return (lowcase not in self.stopwords
                and lowcase not in EX_STOPWORDS
                and lowcase.isalnum())


class WordGenerator:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.morph = MorphProcessor()

    def text_to_words(self, text):
        words = []
        for word_text, word_context in self.tokenizer.tokenize(text):
            lowcase = word_text.lower()
            morph_info = self.morph.process(lowcase)
            words.append(Word(word_text, lowcase, morph_info, word_context))
        return words
