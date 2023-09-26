import sys
from collections import defaultdict
import math
import random
import os
import os.path

"""
COMS W4705 - Natural Language Processing - Summer 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    ng = []
    if n < 1:
        raise Exception('n < 1')
    if n > 1:
        sequence = ['START'] * (n - 1) + sequence + ['STOP']
    else:
        sequence = ['START'] + sequence + ['STOP']

    for i in range(0, len(sequence) - (n - 1)):
        ng.append(tuple(sequence[i:i + n]))
    return ng


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int)  # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.total_num = 0

        for sentence in corpus:
            ng = [get_ngrams(sentence, 1), get_ngrams(sentence, 2), get_ngrams(sentence, 3)]
            # print(ng)
            for index, com in enumerate(ng):
                for word in com:
                    if index == 0:
                        # print(word)
                        self.unigramcounts[word] += 1
                        if word != ('START',):
                            self.total_num += 1
                    if index == 1:
                        self.bigramcounts[word] += 1
                    if index == 2:
                        self.trigramcounts[word] += 1


    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        # if self.bigramcounts[trigram[0:2]] == 0:
        #     return 1/ len(self.lexicon)
        # elif trigram[0:2] == ('START', 'START'):
        #     self.co +=1
        #     return self.raw_bigram_probability(('START', trigram[2]))

        if self.bigramcounts[trigram[0:2]] == 0:

            if trigram[0:2] == ('START', 'START'):
                self.co +=1
                return self.raw_bigram_probability(('START', trigram[2]))
            return 1 / len(self.lexicon)

        return self.trigramcounts[trigram] / self.bigramcounts[trigram[0:2]]

        # try:
        #     return self.trigramcounts[trigram] / self.bigramcounts[trigram[0:2]]
        # except:
        #     return 0.0



    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        try:
            return self.bigramcounts[bigram] / self.unigramcounts[(bigram[0],)]
        except:
            return 0.0

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

        if unigram[0] == 'START':
            return 0.0
        return self.unigramcounts[unigram] / self.total_num

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return None

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """

        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        result = lambda1 * self.raw_trigram_probability(trigram) + \
                lambda2 * self.raw_bigram_probability(trigram[1:3]) + \
                lambda3 * self.raw_unigram_probability((trigram[2],))


        return result

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        tri = get_ngrams(sentence, 3)
        total_pro = 0
        for word in tri:
            # x = self.smoothed_trigram_probability(word)
            # print(word)
            total_pro += math.log2(self.smoothed_trigram_probability(word))
        return total_pro

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        lo = 0
        to = 0
        self.co = 0
        for sentence in corpus:
            lo += self.sentence_logprob(sentence)
            to += (len(sentence)+1)

        result = 2** (-lo/to)
        # print(self.co)
        return result


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if pp1 < pp2:
            correct += 1
        total += 1

    for f in os.listdir(testdir2):
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        if pp1 > pp2:
            correct += 1
        total += 1

    return correct/total


if __name__ == "__main__":
    model = TrigramModel(sys.argv[1])

    # get_ngrams(["natural", "language", "processing"], 1)
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    dd = model.perplexity(corpus_reader(sys.argv[1], model.lexicon))
    print('test p', pp)
    print('train p', dd)

    # Essay scoring experiment:
    acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt',
                                   "hw1_data/ets_toefl_data/train_low.txt",
                                   "hw1_data/ets_toefl_data/test_high",
                                   "hw1_data/ets_toefl_data/test_low")
    print('score',acc)
