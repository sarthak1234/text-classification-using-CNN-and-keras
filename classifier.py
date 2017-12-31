import glob
import os
import json
from keras.models import load_model
from data_helpers import load_data, text_to_vector
import numpy as np
from spacy.en import English
import unicodedata
import traceback
import json
import os
script_dir = os.path.dirname(__file__)

abs_file_path = os.path.join(script_dir, 'datasets/stop_words.txt')
fp = open(abs_file_path)
stop_words=fp.read().split('\n')
fp.close()

# vocab_file = 'vocabulary/vocab_size_33883_epoch_8_time_2017-06-08_13:28:38.json'
# vocab_file = 'vocabulary/vocab_size_33883_epoch_8_time_2017-06-08_13:28:38.json'
vocab_file = 'vocabulary/vocab_size_40065_epoch_5_time_2017-06-21_15:26:33.json'
# model_file = 'model/model_size_33883_epoch_8_time_2017-06-08_13:28:38.h5'
# model_file = 'model/model_size_33883_epoch_8_time_2017-06-08_13:28:38.h5'
model_file = 'model/model_size_40065_epoch_5_time_2017-06-21_15:26:33.h5'

vocabulary = {}
vocabulary_inv = []
parser = English()
model = None

def load_vocabulary():
    """
    Load latest vocabulary file
    """
    global vocabulary
    global vocabulary_inv
    # list_of_vocabs = glob.glob('./vocabulary/*')
    latest_file = os.path.join(script_dir, vocab_file)
    if latest_file:
        # latest_file = max(list_of_vocabs, key=os.path.getctime)
        print "Vocab file: ", latest_file
        fp = open(latest_file, 'rU')
        data = json.loads(fp.read())
        vocabulary = data.get('vocabulary', {})
        vocabulary_inv = data.get('vocabulary_inv', [])
        fp.close()
    return

def myround(a, decimals=1):
     return np.around(a-10**(-(decimals+5)), decimals=decimals)

def get_latest_model():
    list_of_models = glob.glob('./model/*')
    latest_model = None
    if list_of_models:
        latest_model = max(list_of_models, key=os.path.getctime)    
    return latest_model

def load_saved_model():
    global model
    # latest_model = get_latest_model()
    latest_model = os.path.join(script_dir, model_file)
    print "Latest model", latest_model
    if latest_model:
        model = load_model(latest_model)

def classify(sentences):
    global model
    xVal = text_to_vector(sentences, vocabulary)
    yVal = model.predict(xVal)
    yVal = myround(yVal, 2)
    return yVal

def get_verbed_phrase(token):
    s = ''
    for t in token.lefts:
        if t.pos_=='VERB' and t.dep_!='aux' and t.dep_!='auxpass' :
            return s+" "+token.orth_      #for lemmatization
        get_verbed_phrase(t)
    s=s+" "+token.orth_# for lemmatization
    for t in token.rights:
        if t.pos_=='VERB' and t.dep_!='aux' and t.dep_!='auxpass' :
            return s
        get_verbed_phrase(t)
    return s

def lemmatized(text):     #returns lemmatized form of string
    string = parser(unicode(text))
    lem=""
    for token in string :
        if token not in stop_words:
            lem = lem+" "+token.lemma_
    return lem

def remove_stop_words(string):
    words = string.split(" ")
    for w in words :
        if w in stop_words :
            words.remove(w)
    string = ""
    for w in words :
        string = string+" "+w
    return string

def get_phrases_sentiment(phrases):
    if not len(phrases):
        return []
    sentiments = classify([lemmatized(phrase) for phrase in phrases])
    review_phrases = []
    for (i, phrase) in enumerate(phrases):
        if sentiments[i][0] > sentiments[i][1]:
            review_phrases.append({'text': phrase, 'classifier_polarity': -1})
        else:
            review_phrases.append({'text': phrase, 'classifier_polarity': 1})
    return review_phrases

load_saved_model()
load_vocabulary()

if __name__ == '__main__':
    # print "Hello World !"
    test_phrases = [
    "you won't regret",
    "happy to find out that the buffet does not make a hole",
    "helping attitude",
    "well maintained space round the clock service available",
    "good support staff service and attitude"
    ]
    res = get_phrases_sentiment(test_phrases)
    print json.dumps(res, indent=2)
