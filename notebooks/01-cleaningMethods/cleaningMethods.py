from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re, nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd

lemmatizer = WordNetLemmatizer()

from nlpHelpers import Contraction_Dictionary, stop_words, PosList, PosMapper, negations

# This method normalizes the text into a coherent format for matching
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.lower() # Convert to lowercase
    df[text_field] = df[text_field].str.replace('http','') # removing urls is useful to make vocabulary small as possible
    df[text_field] = df[text_field].str.replace('com', '') # same as above.
    df[text_field] = df[text_field].str.replace('[0-9]', '')
    df[text_field] = df[text_field].str.replace(r"@", "at") #  replacing at sign for a word
    df[text_field] = df[text_field].str.replace(".", " ")
    df[text_field] = df[text_field].str.replace(",", " ")
    df[text_field] = df[text_field].str.replace("-", " ")
    df[text_field] = df[text_field].str.replace("(", " ")
    df[text_field] = df[text_field].str.replace(")", " ")
    df[text_field] = df[text_field].str.replace('"', " ")
    df[text_field] = df[text_field].str.replace("?", "")
    df[text_field] = df[text_field].str.replace("!", "")
    df[text_field] = df[text_field].str.replace('_',' ')
    df[text_field] = df[text_field].str.replace("`", '/')
    df[text_field] = df[text_field].str.lstrip(' ')
    return df


def expand_contractions(text, contraction_mapping=Contraction_Dictionary):
# This method expands all contractions to their original format
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def doubleLetterRemoval(string_object):
	tokens = nltk.word_tokenize(string_object)
	listOfTokens = []
	for token in tokens:
		pattern = re.compile(r"(.)\1{2,}")
		word = pattern.sub(r"\1\1", token)
		if len(word) > 2:
			listOfTokens.append(word)
	return listOfTokens

def stopwordRemoval(string_object):
	tokens = nltk.word_tokenize(string_object)
	listOfTokens = []
	for token in tokens:
		if not token in stop_words:
			listOfTokens.append(token)
	return listOfTokens

#Spell checker test for lists of list of tokens
def spellChecker(string_object):
	tokens = nltk.word_tokenize(string_object)
	checkedTokens = []
	import enchant # This is a python spell checker form the PyEnchant library 
	d = enchant.Dict("en_Uk")
	for token in tokens:
		if d.check(token) == True:
			tokens.append(token)
	return tokens

def lemma(string_object):
	tokens = nltk.word_tokenize(string_object)
	posTupples = nltk.pos_tag(tokens)
	text = [lemmatizer.lemmatize(k[0], pos=PosMapper.get(k[1])) if k[1] in PosList else k[0] for k in posTupples]
	return text

### Replacing words with unambiguous antonyms by using WordNet.
from nltk.corpus import wordnet as wn
class word_antonym_replacer(object):

    def replace(self, word, pos=None):
        antonyms = set()
        for syn in wn.synsets(word, pos=pos): # pos argument which lets you constrain the part of speech of the word
                                            # Synset: a set of synonyms that share a common meaning.
            for lemma in syn.lemmas(): # Each synset contains one or more lemmas, which represent a specific sense of a specific word.
                for antonym in lemma.antonyms(): # An antonym is a word that has the opposite meaning of another word.
                    antonyms.add(antonym.name())
        if len(antonyms) == 1: # If there is only 1 antonym replacement option replace that word.
            return antonyms.pop() # removes and returns last value from the list or the given index value
        else:
            return None
    def wordRetrieval(self, token):
        posTag = nltk.word_tokenize(token)
        posTag = nltk.pos_tag(posTag)
        print(posTag)
        score = 0
        for tag in posTag:
            if tag[1] in adjList:
                word = tag[0]
                score = 1
            else:
                score = 0
        return score
    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word in negations and i+1 < l: # here we would sub in a list of negation terms:
                test = self.wordRetrieval(sent[i+1])
                if test == 1:
                    print(test)
                    ant = self.replace(sent[i+1])
                    if ant:
                        words.append(ant)
                        i += 2
                        continue
            words.append(word)
            i += 1
        return words
