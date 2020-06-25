from nltk.corpus import wordnet


Contraction_Dictionary = {
    "ain/t": "is not", "aren/t": "are not",'cannot':"can not", "can/t": "can not", "can/t/ve": "can not have", "cause": "because", "could/ve": "could have",
    "couldn/t": "could not", "couldn/t/ve": "could not have", "didn/t": "did not", "doesn/t": "does not", "don/t": "do not", "hadn/t": "had not",
    "hadn/t/ve": "had not have", "hasn/t": "has not", "haven/t": "have not", "he/d": "he would", "he/d/ve": "he would have", "he/ll": "he will",
    "he/ll/ve": "he he will have", "he/s": "he is", "how/d": "how did", "how/d/y": "how do you", "how/ll": "how will", "how/s": "how is",
    "I/d": "I would", "I/d/ve": "I would have", "I/ll": "I will", "I/ll/ve": "I will have", "I/m": "I am", "I/ve": "I have", "i/d": "i would",
    "i/d/ve": "i would have", "i/ll": "i will", "i/ll/ve": "i will have", "i/m": "i am", "i/ve": "i have", "isn/t": "is not", "it/d": "it would",
    "it/d/ve": "it would have", "it/ll": "it will", "it/ll/ve": "it will have", "it/s": "it is", "let/s": "let us", "ma/am": "madam", "mayn/t": "may not",
    "might/ve": "might have", "mightn/t": "might not", "mightn/t/ve": "might not have", "must/ve": "must have", "mustn/t": "must not", "mustn/t/ve": "must not have",
    "needn/t": "need not", "needn/t/ve": "need not have", "o/clock": "of the clock", "oughtn/t": "ought not", "oughtn/t/ve": "ought not have", "shan/t": "shall not",
    "sha/n/t": "shall not", "shan/t/ve": "shall not have", "she/d": "she would", "she/d/ve": "she would have", "she/ll": "she will", "she/ll/ve": "she will have",
    "she/s": "she is", "should/ve": "should have", "shouldn/t": "should not", "shouldn/t/ve": "should not have", "so/ve": "so have", "so/s": "so as",
    "that/d": "that would", "that/d/ve": "that would have", "that/s": "that is", "there/d": "there would", "there/d/ve": "there would have",
    "there/s": "there is", "they/d": "they would", "they/d/ve": "they would have", "they/ll": "they will", "they/ll/ve": "they will have", "they/re": "they are",
    "they/ve": "they have", "to/ve": "to have", "wasn/t": "was not", "we/d": "we would", "we/d/ve": "we would have", "we/ll": "we will", "we/ll/ve": "we will have", 
    "we/re": "we are", "we/ve": "we have", "weren/t": "were not", "what/ll": "what will", "what/ll/ve": "what will have","what/re": "what are", "what/s": "what is", 
    "what/ve": "what have", "when/s": "when is", "when/ve": "when have", "where/d": "where did", "where/s": "where is", "where/ve": "where have",
    "who/ll": "who will", "who/ll/ve": "who will have", "who/s": "who is", "who/ve": "who have", "why/s": "why is", "why/ve": "why have", "will/ve": "will have", 
    "won/t": "will not","won/t/ve": "will not have", "would/ve": "would have", "wouldn/t": "would not", "wouldn/t/ve": "would not have", "y/all": "you all",
    "y/all/d": "you all would", "y/all/d/ve": "you all would have", "y/all/re": "you all are", "y/all/ve": "you all have", "you/d": "you would",
    "you/d/ve": "you would have", "you/ll": "you will", "you/ll/ve": "you will have", "you/re": "you are", "you/ve": "you have",
    "aint": "is not", "arent": "are not","cannot":"can not", "cant": "can not", "cantve": "can not have", "cause": "because", "couldve": "could have",
    "couldnt": "could not", "couldntve": "could not have", "didnt": "did not", "doesnt": "does not", "dont": "do not", "hadnt": "had not",
    "hadntve": "had not have", "hasnt": "has not", "havent": "have not", "hed": "he would", "hedve": "he would have", "hell": "he will",
    "hellve": "he he will have", "hes": "he is", "howd": "how did", "howdy": "how do you", "howll": "how will", "hows": "how is",
    "Id": "I would", "Idve": "I would have", "Ill": "I will", "Illve": "I will have", "Im": "I am", "Ive": "I have", "id": "i would",
    "idve": "i would have", "ill": "i will", "illve": "i will have", "im": "i am", "ive": "i have", "isnt": "is not", "itd": "it would",
    "itdve": "it would have", "itll": "it will", "itllve": "it will have", "its": "it is", "lets": "let us", "maam": "madam", "maynt": "may not",
    "mightve": "might have", "mightnt": "might not", "mightntve": "might not have", "mustve": "must have", "mustnt": "must not", "mustntve": "must not have",
    "neednt": "need not", "needntve": "need not have", "oclock": "of the clock", "oughtnt": "ought not", "oughtntve": "ought not have", "shant": "shall not",
    "shant": "shall not", "shantve": "shall not have", "shed": "she would", "shedve": "she would have", "shell": "she will", "shellve": "she will have",
    "shes": "she is", "shouldve": "should have", "shouldnt": "should not", "shouldntve": "should not have", "sove": "so have", "sos": "so as",
    "thatd": "that would", "thatdve": "that would have", "thats": "that is", "thered": "there would", "theredve": "there would have",
    "theres": "there is", "theyd": "they would", "theydve": "they would have", "theyll": "they will", "theyllve": "they will have", "theyre": "they are",
    "theyve": "they have", "tove": "to have", "wasnt": "was not", "wed": "we would", "wedve": "we would have", "well": "we will", "wellve": "we will have", 
    "were": "we are", "weve": "we have", "werent": "were not", "whatll": "what will", "whatllve": "what will have","whatre": "what are", "whats": "what is", 
    "whatve": "what have", "whens": "when is", "whenve": "when have", "whered": "where did", "wheres": "where is", "whereve": "where have",
    "wholl": "who will", "whollve": "who will have", "whos": "who is", "whove": "who have", "whys": "why is", "whyve": "why have", "willve": "will have", 
    "wont": "will not","wontve": "will not have", "wouldve": "would have", "wouldnt": "would not", "wouldntve": "would not have", "yall": "you all",
    "yalld": "you all would", "yalldve": "you all would have", "yallre": "you all are", "yallve": "you all have", "youd": "you would",
    "youdve": "you would have", "youll": "you will", "youllve": "you will have", "youre": "you are", "youve": "you have"
}

stop_words =['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself',
            'yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself',
            'they','them','their','theirs','themselves','what','which','who','whom','this','that',
            'these','those','am','is','are','was','were','be','been','being','have','has','had',
            'having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
            'until','while','of','at','by','for','with','about','against','between','into','through',
            'during','before','after','above','below','to','from','up','down','in','out','on','off',
            'over','under','again','further','then','once','here','there','when','where','why','how',
            'all','any','both','each','few','more','most','other','some','such',
            'only','own','same','so','than','too','very','can','will','just','should',
            'now','uses','use','using','used','one','also']


PosList =["JJ","JJR","JJS","NN","NNS","NNP","NNPS","RB",
          "RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"]

PosMapper = {
"JJ": wordnet.ADJ,
"JJR": wordnet.ADJ,
"JJS": wordnet.ADJ,
"NN": wordnet.NOUN,
"NNS": wordnet.NOUN,
"NNP": wordnet.NOUN,
"NNPS": wordnet.NOUN,
"RB": wordnet.ADV,
"RBR": wordnet.ADV,
"RBS": wordnet.ADV,
"VB": wordnet.VERB,
"VBD": wordnet.VERB,
"VBG": wordnet.VERB,
"VBN": wordnet.VERB,
"VBP": wordnet.VERB,
"VBZ": wordnet.VERB}

negations = ['no', 'not','never','none', 'nobody','nothing','neither','nor','nowhere','nor','no one']