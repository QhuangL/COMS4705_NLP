"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
import math
from collections import defaultdict
from math import fsum
import string

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        for ind, lhs_sen in self.lhs_to_rules.items():
            # print(ind)
            # print(lhs)
            sum_p = []
            for lhs in lhs_sen:
                sum_p.append(lhs[2])
                rhs = lhs[1]
                if len(rhs) == 2 and rhs[0].isupper() and rhs[1].isupper():
                    pass
                elif len(rhs) == 1 and (rhs[0].islower() or rhs[0] in string.punctuation or rhs[0].isdigit()):
                    pass
                else:
                    return False

            if math.isclose(fsum(sum_p), 1.0):
                return True

        return False 


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        
