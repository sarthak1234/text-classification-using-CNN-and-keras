import numpy as np
import re
import itertools
from collections import Counter
import random
from spacy.en import English
import unicodedata
import os.path
import os
script_dir = os.path.dirname(__file__)

sequence_length = 100
parser = English()
abs_file_path = os.path.join(script_dir, 'datasets/stop_words.txt')
fp = open(abs_file_path)
stop_words=fp.read().split('\n')
fp.close()

def lemmatize_data(file_path, save_path):
    """
    Lemmatize sentences from file_path and save it to save_path
    """
    fp_read = open(file_path, 'rU')
    fp_save = open(save_path, 'w')
    lines = fp_read.readlines()
    for line in lines:
        tokens = parser(unicode(line.strip()))
        lemmatized_text = ''
        for token in tokens:
            if token not in stop_words:
                lemmatized_text = lemmatized_text+" "+unicode(token.lemma_)
        fp_save.write(lemmatized_text+'\n')
    fp_read.close()
    fp_save.close()

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    print 'Load data from files.....'

    # if not os.path.isfile("./data/lemmatized_positive.txt"):
    #     print "Lemmatized positive text not found !"
    #     lemmatize_data('./data/corrected_positive.txt', "./data/lemmatized_positive.txt")
    #     print "Lemmatized file created !"

    # if not os.path.isfile("./data/lemmatized_negative.txt"):
    #     print "Lemmatized negative text not found !"
    #     lemmatize_data('./data/corrected_negative.txt', './data/lemmatized_negative.txt')
    #     print "Lemmatized file created !"
    abs_file_path = os.path.join(script_dir, "datasets/lemmatized_positive_aspect_opinion.txt")
    positive_examples = list(open(abs_file_path, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    random.shuffle(positive_examples)
    abs_file_path = os.path.join(script_dir, "datasets/lemmatized_negative_aspect_opinion.txt")
    negative_examples = list(open(abs_file_path, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    random.shuffle(negative_examples)
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary.get(word, 1) for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def text_to_vector(sentences, vocabulary):
    """
    Convert list of text into list of padded id vectors
    """
    x_text = [clean_str(sent) for sent in sentences]
    x_text = [s.split(" ") for s in x_text]
    sentences_padded = pad_sentences(x_text)
    x, y = build_input_data(sentences_padded, [], vocabulary)
    return x

if __name__ == '__main__':
    print "Hello World !"
    # sentences = [
    #     "surprised the room didn't have a kettle",
    #     "room was spacious",
    #     "non cooperative staff",
    #     "staff was very helpful"
    # ]
    # fp = open('./data/lemmatized_positive.txt')
    # pos_reviews = [line.strip() for line in fp.readlines()]
    # fp.close()
    # fp = open('./data/lemmatized_negative.txt')
    # neg_reviews = [line.strip() for line in fp.readlines()]
    # fp.close()
    # sentences = pos_reviews + neg_reviews

    # x_text = [clean_str(sent) for sent in sentences]
    # x_text = [s.split(" ") for s in x_text]
    # sentences_padded = pad_sentences(x_text)
    # vocab, vocab_inv = build_vocab(sentences_padded)

    # print len(vocab.keys())
    # # data = text_to_vector(sentences, vocab)