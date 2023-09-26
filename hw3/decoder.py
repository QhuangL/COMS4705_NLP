from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def action_permitted(self, action, state):
        if action[0] == 'shift':
            if len(state.buffer) > 1:
                return True
            elif len(state.buffer) == 1 and len(state.stack) == 0:
                return True
            else:
                return False

        elif action[0] == 'left_arc':
            if len(state.stack) > 0:
                if state.stack[-1] != 0:
                    return True
            else:
                return False

        elif action[0] == 'right_arc':
            if len(state.stack) > 0:
                return True
            else:
                return False
    def sort_output(self, smax_output):
        action_prob_tuples = []
        for i in range(0, len(smax_output)):
            action_prob_tuples.append((self.output_labels[i], smax_output[i]))
        return sorted(action_prob_tuples, key=lambda x: x[1], reverse=True)

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        it = 0
        while state.buffer: 
            input = self.extractor.get_input_representation(words, pos, state)
            smax_output = self.model.predict(input.reshape(1, -1))
            sorted_actions = self.sort_output(smax_output[0])

            for i in range(0, len(sorted_actions)):
                action = sorted_actions[i][0]
                if self.action_permitted(action, state):
                    if action[0] == 'shift':
                        state.shift()
                        # For left and right arc, supply the relationship type
                    elif action[0] == 'left_arc':
                        state.left_arc(action[1])
                    elif action[0] == 'right_arc':
                        state.right_arc(action[1])
                    break
            it += 1
        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
