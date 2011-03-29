"""
Methods for preprocessing text.

@author: Kjetil Valle <kjetilva@stud.ntnu.no>
"""

import nltk, re, os
from tempfile import mkstemp
from shutil import move

#####
##
## Tokenizers
##
#####

def tokenize_tokens(text):
    """Return list of tokens for text string input"""
    return nltk.tokenize.word_tokenize(text)

def tokenize_sentences(text):
    """Returns a list of sentences (as strings) from the input text"""
    return nltk.tokenize.sent_tokenize(text)

#####
##
## Token processing
##
## The following functions operate on lists of tokens.
##
#####

def fold_case(tokens):
    return [tok.lower() for tok in tokens]

stopwords = nltk.corpus.stopwords.words('english')
def remove_stop_words(tokens):
    return [t for t in tokens if t.lower() not in stopwords]

def is_stop_word(token):
    return token.lower() in stopwords

stemmer = nltk.PorterStemmer()
def stem(tokens):
    return [stemmer.stem(t) for t in tokens]

def filter_tokens(tokens, min_size=0, special_chars=False):
    """
    @param min_size: remove any token with length below threshold
    @param special_chars: remove tokens containing special chars
    """
    if min_size>0:
        tokens = [t for t in tokens if len(t) >= min_size]
    if special_chars:
        tokens = [t for t in tokens if re.search('[^a-zA-Z-]',t)==None]
    return tokens

#####
##
## Processing methods
##
#####

def preprocess_token(token, do_stop_word_removal=True, do_stemming=True, fold=True, specials=True, min_size=3):
    """Perform preprocessing on a single token."""
    if fold: token = fold_case([token])[0]
    if do_stop_word_removal and is_stop_word(token): return None
    if do_stemming: token = stem([token])[0]
    filt = filter_tokens([token], min_size, specials)
    if len(filt)==0: return None
    else: return filt[0]

def preprocess_text(text, do_stop_word_removal=True, do_stemming=True, fold=True, specials=True, min_size=3):
    """Perform preprocessing steps on the input text."""
    ts = tokenize_tokens(text)
    ts = filter_tokens(ts, min_size=min_size,special_chars=specials)
    if fold:
        ts = fold_case(ts)
    if do_stop_word_removal:
        ts = remove_stop_words(ts)
    if do_stemming:
        ts = stem(ts)
    return ts

def extract_dependencies(text):
    """
    Creates a dictionary with dependency information about the text.

    Some sentences may be too long for the parser to handle, leading to
    exhaustion of the java heap space. If this happens the sentence is
    skipped and its dependencies omitted.
    """
    import stanford_parser_pype as spp
    import jpype

    results = {}
    sentences = tokenize_sentences(text)
    for s in sentences:
        try:
            pos, tree, dependencies = spp.parse(s)
        except jpype._jexception.JavaException as e:
            print '! Sentence too long, skipping it;\n!', e
            continue
        for d in dependencies:
            dep_type = d.reln().getShortName()
            dep_type = dep_type
            if dep_type not in results.keys():
                results[dep_type] = []
            gov = d.gov()
            dep = d.dep()
            gov_tag = pos[gov.index()-1].tag()
            dep_tag = pos[dep.index()-1].tag()
            gov_word = gov.value()
            dep_word = dep.value()
            results[dep_type].append( ((gov_word,gov_tag),(dep_word,dep_tag)) )
    return results

#####
##
## Tests
##
## Simple tests of functions in this module.
##
#####

def test_tokenize_tokens():
    text = "This is sentence one."
    tok = tokenize_tokens(text)
    assert(tok==['This', 'is', 'sentence', 'one', '.'])

def test_tokenize_sentences():
    text = "This is sentence one. And this, my friend, is sentence two. And this here -- with an paranthetical dash sentence shot in -- is the last sentence."
    ss = tokenize_sentences(text)
    assert(ss[0]=="This is sentence one.")
    assert(ss[1]=="And this, my friend, is sentence two.")
    assert(ss[2]=="And this here -- with an paranthetical dash sentence shot in -- is the last sentence.")

def test_stem():
    pre = ['fishing', 'fishes', 'cakes']
    post = ['fish', 'fish', 'cake']
    assert(stem(pre)==post)

def test_fold_case():
    pre = ['Hi!', 'MiXeD', 'caSE']
    post = ['hi!', 'mixed', 'case']
    assert(fold_case(pre)==post)

def test_remove_stop_words():
    pre = ['The', 'cake', 'is', 'a', 'lie']
    post = ['cake', 'lie']
    assert(remove_stop_words(pre)==post)

def test_is_stop_word():
    assert(is_stop_word('the')==True)
    assert(is_stop_word('The')==True)
    assert(is_stop_word('cake')==False)

def test_filter_tokens():
    pre = ['ab', 'abc', '1abc', '12']
    assert(filter_tokens(pre, min_size=3)==['abc', '1abc'])
    assert(filter_tokens(pre, special_chars=True)==['ab', 'abc'])
    assert(filter_tokens(pre, min_size=3, special_chars=True)==['abc'])

def test_preprocess_text():
    text = "A review of the radar data indicated that the aircraft approached the Liverpool airport from the east, turned south across runway 25/07 and joined the circuit left-hand downwind for runway 25."
    processed = ['review', 'radar', 'data', 'indic', 'aircraft', 'approach', 'liverpool', 'airport', 'east', 'turn', 'south', 'across', 'runway', 'join', 'circuit', 'downwind', 'runway']
    assert(preprocess_text(text)==processed)

def run_tests():
    test_tokenize_sentences()
    test_tokenize_tokens()
    test_stem()
    test_fold_case()
    test_remove_stop_words()
    test_filter_tokens()
    test_preprocess_text()
    test_is_stop_word()
    print "ok"

# -----

def tmp_stop_words(word):
    print word in stopwords
    print is_stop_word(word)

def tmp_stem(word):
    print stem([word])[0]

if __name__ == "__main__":
    #~ run_tests()
    word = 'the'
    tmp_stop_words(word)
    tmp_stem(word)
