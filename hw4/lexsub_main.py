#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

from collections import Counter
import string

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split()

def lower(words):
    lowered_words = []
    for word in words:
        lowered_words.append(word.lower())

    return lowered_words

def remove_punctuation(sentence):
    cleaned_sentence = []
    for word in sentence:
        if word not in string.punctuation:
            cleaned_sentence.append(word)
    return cleaned_sentence

def remove_stopwords(words):
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)

    return new_words

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    syno = set([])
    lexs = wn.lemmas(lemma, pos= pos)

    for i in lexs:
        syn = i.synset()

        for sing in syn.lemmas():
            sing_name = sing.name()

            if sing_name != lemma:

                if len(sing_name.split('_')) > 1:

                    sing_name = sing_name.replace('_', ' ')
                syno.add(sing_name)

    return syno

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # replace for part 2
    syno = Counter()
    lexs = wn.lemmas(context.lemma, pos=context.pos)

    for i in lexs:
        syn = i.synset()

        for sing in syn.lemmas():
            sing_name = sing.name()

            if sing_name != context.lemma:

                if len(sing_name.split('_')) > 1:
                    sing_name = sing_name.replace('_', ' ')
                syno[sing_name] += sing.count()

    return syno.most_common(1)[0][0]

def get_most_frequent_lexeme(synset, input_lemma):
    max_lexeme = None
    max_count = 0

    # Get the lemmas from the synset
    for lex in synset.lemmas():
        if lex.name() != input_lemma:
            if lex.count() > max_count:
                max_lexeme = lex
                max_count = lex.count()

    if max_count == 0:
        return synset.lemmas()[0]

    return max_lexeme

def get_most_frequent_synset(synset_overlap_list, input_lemma):
    max_count = 0
    most_frequent_synset = None

    for tup in synset_overlap_list:
        synset = tup[0]
        lexemes = synset.lemmas()

        proceed = True
        if len(lexemes) == 1:
            if lexemes[0].name() == input_lemma:
                proceed = False

        if proceed:
            synset = tup[0]
            count = 0

            for lexeme in synset.lemmas():
                count += lexeme.count()

            if count > max_count:
                most_frequent_synset = synset
                max_count = count

    return most_frequent_synset

def compute_overlap(cleaned_full_context, sense):
    overlap = 0
    cleaned_full_context_set = set(cleaned_full_context)

    raw_definition = sense.definition()
    definition = tokenize(raw_definition)
    examples = sense.examples()

    cleaned_definition = remove_stopwords(definition)
    cleaned_examples = remove_stopwords(examples)

    overlap += len(cleaned_full_context_set.intersection(set(cleaned_definition)))
    overlap += len(cleaned_full_context_set.intersection(set(cleaned_examples)))

    hypernyms = sense.hypernyms()
    for hypernym in hypernyms:
        raw_definition = hypernym.definition()
        definition = tokenize(raw_definition)
        examples = hypernym.examples()

        cleaned_definition = remove_stopwords(definition)
        cleaned_examples = remove_stopwords(examples)

        overlap += len(cleaned_full_context_set.intersection(set(cleaned_definition)))
        overlap += len(cleaned_full_context_set.intersection(set(cleaned_examples)))

    if overlap > 0:
        pass

    return overlap

def get_cleaned_full_context(context, window_size=-1, pad='left'):
    left_context = context.left_context
    right_context = context.right_context

    left_context = remove_punctuation(left_context)
    left_context = lower(left_context)
    left_context = remove_stopwords(left_context)

    right_context = remove_punctuation(right_context)
    right_context = lower(right_context)
    right_context = remove_stopwords(right_context)

    if window_size > 0:
        left_window_size = window_size // 2

        if window_size % 2 == 1:
            if pad == 'left':
                left_window_size = left_window_size + 1

        right_window_size = window_size - left_window_size

        if len(left_context) > left_window_size:
            left_context = left_context[len(left_context) - left_window_size:len(left_context)]

        if len(right_context) > right_window_size:
            right_context = right_context[0:right_window_size]

    full_context = left_context + right_context

    return full_context


def wn_simple_lesk_predictor(context : Context) -> str:
    # replace for part 3
    cleaned_context = get_cleaned_full_context(context)

    synset_overlap_list = []

    # Iterate over synsets
    for synset in wn.synsets(context.lemma, context.pos):
        lexemes = synset.lemmas()
        proceed = True
        if len(lexemes) == 1:
            if lexemes[0].name() == context.lemma:
                proceed = False

        if proceed:
            overlap = compute_overlap(cleaned_context, synset)

            synset_overlap_list.append((synset, overlap))

    synset_overlap_list = sorted(synset_overlap_list, key=lambda x: x[1], reverse=True)

    best_tup = synset_overlap_list[0]
    best_synset = best_tup[0]
    best_overlap = best_tup[1]

    if best_overlap == 0:
        best_sense = get_most_frequent_synset(synset_overlap_list, context.lemma)

    most_frequent_lexeme = get_most_frequent_lexeme(best_synset, context.lemma)

    lemma_name = most_frequent_lexeme.name()

    # Check if lemma contains multiple words
    if len(lemma_name.split('_')) > 1:
        lemma_name = lemma_name.replace('_', ' ')

    return lemma_name

   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        # replace for part 4
        syno = list(get_candidates(context.lemma, context.pos))

        considered_synonyms = []
        for sy in syno:
            if sy in self.model:
                considered_synonyms.append(sy)

        return self.model.most_similar_to_given(context.lemma, considered_synonyms)



class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # replace for part 5
        candidates = get_candidates(context.lemma, context.pos)
        input = " ".join(map(str, context.left_context))
        input += " " + context.lemma + " "
        input += " ".join(map(str, context.right_context))
        input_toks = self.tokenizer.tokenize(input)
        idx = input_toks.index(context.lemma)
        input_toks[idx] = '[MASK]'
        input_toks = self.tokenizer.encode(input_toks)
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat, verbose= False)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][idx + 1])[::-1]  # Sort in increasing order
        words = self.tokenizer.convert_ids_to_tokens(best_words)
        for w in words:
            if w in candidates:
                return w

class customPredictor(object):
    def __init__(self):
        self.Bert = BertPredictor()
        self.W2VS = Word2VecSubst('GoogleNews-vectors-negative300.bin.gz')

    def predict(self, contect: Context) -> str:
        # Choose word with highest count from words predicted (by best predictors)
        p1 = wn_simple_lesk_predictor(context)
        p2 = self.Bert.predict(context)
        p3 = self.W2VS.predict_nearest(context)
        pred = [p1, p2, p3]
        return max(pred, key=pred.count)
    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    predictor1 = BertPredictor()
    predictor2 = customPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        # prediction = smurf_predictor(context)
        # prediction = wn_frequency_predictor(context)
        # prediction = wn_simple_lesk_predictor(context)
        # prediction = predictor.predict_nearest(context)
        # prediction = predictor1.predict(context)
        prediction = predictor2.predict(context) # compare three methods and find the best policy
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
