"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        n = len(tokens)

        #initialize table
        p_table = {}
        for i in range(n+1):
            for j in range(i+1, n+1):
                p_table[(i, j,)] = {}

        #main loop
        for i in range(n):
            rules = self.grammar.rhs_to_rules[(tokens[i],)]
            for rule in rules:
                nt = rule[0]
                p_table[(i, i+1,)][nt] = ()

        for length in range(2, n+1):
            for i in range(n+1-length):
                j = i + length

                for k in range(i+1, j):
                    some_nt_1 = p_table[(i, k, )]
                    some_nt_2 = p_table[(k, j, )]

                    for nt_1 in some_nt_1:
                        for nt_2 in some_nt_2:
                            com = (nt_1, nt_2, )

                            if com in self.grammar.rhs_to_rules:
                                rhs_to_rules = self.grammar.rhs_to_rules[com]

                                for rule in rhs_to_rules:
                                    new_nt = rule[0]
                                    p_table[(i, j,)][new_nt] = ()

        return 'TOP' in p_table[(0, len(tokens), )]
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        p_table = {}
        probs = {}

        n = len(tokens)
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                p_table[(i, j, )] = {}
                probs[(i, j, )] = {}

        # main loop
        for i in range(n):
            rules = self.grammar.rhs_to_rules[(tokens[i], )]
            for rule in rules:
                nt = rule[0]
                pro = math.log2(rule[2])

                p_table[(i, i + 1, )][nt] = tokens[i]
                probs[(i, i + 1, )][nt] = pro

        for length in range(2, n + 1):
            for i in range(n + 1 - length):
                j = i + length

                for k in range(i + 1, j):
                    some_nt_1 = p_table[(i, k, )]
                    some_nt_2 = p_table[(k, j, )]

                    for nt_1 in some_nt_1:
                        for nt_2 in some_nt_2:
                            com = (nt_1, nt_2, )

                            if com in self.grammar.rhs_to_rules:
                                rhs_to_rules = self.grammar.rhs_to_rules[com]

                                for rule in rhs_to_rules:
                                    new_nt = rule[0]

                                    pro_tree1 = probs[(i, k, )][nt_1]
                                    pro_tree2 = probs[(k, j, )][nt_2]
                                    pro_num = math.log2(rule[2])
                                    new_prob = pro_tree1 + pro_tree2 + pro_num

                                    bp_1 = (com[0], i, k, )
                                    bp_2 = (com[1], k, j, )

                                    if new_nt not in p_table[(i, j, )]:

                                        p_table[(i, j,)][new_nt] = (bp_1, bp_2, )

                                        # Assign the probability as well
                                        probs[(i, j,)][new_nt] = new_prob
                                    else:
                                        # Check if current prob is bigger than previous prob
                                        if new_prob > probs[(i, j, )][new_nt]:
                                            p_table[(i, j, )][new_nt] = (bp_1, bp_2)

                                            probs[(i, j, )][new_nt] = new_prob

        return p_table, probs


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if isinstance(chart[(i, j)][nt], str):
        return (nt, chart[(i, j)][nt], )
    else:

        left_bp = chart[(i, j)][nt][0]
        right_bp = chart[(i, j)][nt][1]

        left_nt = left_bp[0]
        right_nt = right_bp[0]

        left_i = left_bp[1]
        right_i = right_bp[1]

        left_j = left_bp[2]
        right_j = right_bp[2]

        return ((nt, get_tree(chart, left_i, left_j, left_nt), get_tree(chart, right_i, right_j, right_nt), ))


       
if __name__ == "__main__":
    
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        
