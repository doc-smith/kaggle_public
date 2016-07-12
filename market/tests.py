from hits import generate_hits
from words import WordGenerator
from relevance import ExactMatch, PhraseMatch

class TrackerBase:

    def __init__(self):
        self.wg = WordGenerator()

    def calculate(self, query, text, tracker):
        query = self.wg.text_to_words(query)
        text = self.wg.text_to_words(text)
        hits = generate_hits(query, text)
        tracker.new_doc(query)
        for hit in hits:
            tracker.add_hit(hit)
        return tracker.calculate()
     

class TestExactMatch(TrackerBase):

    def testGoodMatch(self):
        query = 'iphone 5'
        text = 'Iphone 5 sucks.'
        assert self.calculate(query, text, ExactMatch()) == 1

        query = '5 iphone'
        text = 'Iphone 5 sucks.'
        assert self.calculate(query, text, ExactMatch()) == 1

        query = 'iphone sucks'
        text = 'Iphone, sucks.'
        assert self.calculate(query, text, ExactMatch()) == 1

        query = 'iphone sucks'
        text = 'Iphone. Sucks.'
        assert self.calculate(query, text, ExactMatch()) == 1

    def testBadMatch(self):
        query = 'iphone 5'
        text = 'Iphone sucks.'
        assert self.calculate(query, text, ExactMatch()) == 0

        query = 'iphone 5 suck'
        text = 'Iphone 5 sucks.'
        assert self.calculate(query, text, ExactMatch()) == 0

class TestPhraseMatch(TrackerBase):
    
    def testGoodMatch(self):
        query = 'iphone sucks'
        text = 'Iphone sucks.'
        assert self.calculate(query, text, PhraseMatch()) == 1

    def testGoodMatch(self):
        query = 'sucks iphone'
        text = 'Iphone sucks.'
        assert self.calculate(query, text, PhraseMatch()) == 0


class TestWordTokenizator:

    def setUp(self):
        self.wg = WordGenerator()

    def testSaveNumbers(self):
        words = self.wg.text_to_words("Iphone 5")
        assert len(words) == 2
        assert words[1].text == '5'
        assert words[1].lowcase == '5'
        assert words[1].morph_info.lemma == '5'

    def testSaveModelName(self):
        return # TODO: Fix it
        words = self.wg.text_to_words("Iphone 5+")
        assert len(words) == 2
        assert words[1].text == '5+'
        assert words[1].lowcase == '5+'
        assert words[1].morph_info.lemma == '5+'

class TestSentencesBreaks():

    def setUp(self):
        self.wg = WordGenerator()

    def testOneSentence(self):
        words = self.wg.text_to_words("One Two.")
        assert len(words) == 2
        assert words[0].context.sentence_index == words[1].context.sentence_index

        words = self.wg.text_to_words("One, Two.")
        assert len(words) == 2
        assert words[0].context.sentence_index == words[1].context.sentence_index

        words = self.wg.text_to_words("One - Two.")
        assert len(words) == 2
        assert words[0].context.sentence_index == words[1].context.sentence_index

    def testSeveralSentences(self):
        words = self.wg.text_to_words("One. Two.")
        assert len(words) == 2
        assert words[0].context.sentence_index + 1 == words[1].context.sentence_index

        words = self.wg.text_to_words("One? Two.")
        assert len(words) == 2
        assert words[0].context.sentence_index + 1 == words[1].context.sentence_index

        words = self.wg.text_to_words("One! Two.")
        assert len(words) == 2
        assert words[0].context.sentence_index + 1 == words[1].context.sentence_index

