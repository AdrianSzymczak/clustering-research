import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

pos_tags_translation = {
    'NN':wordnet.NOUN,
    'JJ':wordnet.ADJ,
    'VB':wordnet.VERB,
    'RB':wordnet.ADV
}

lemmatizer = WordNetLemmatizer()

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

def part_of_speech(tokens):
    pos_tokens = nltk.pos_tag(tokens)
    return pos_tokens

def filter_irrelevant_tokens(pos_tokens):

    stop = set(stopwords.words('english'))

    pos_tokens_filtered = []

    for word, pos in pos_tokens:
        pos_simple = pos[:2]
        if pos_simple in pos_tags_translation.keys() and word.lower() not in stop:
            pos_tokens_filtered.append((word, pos))

    return pos_tokens_filtered

def lemmatize(pos_tokens):

    lemmatized = []

    for word, pos in pos_tokens:
        pos_simple = pos[:2]
        if pos_simple in pos_tags_translation.keys():
            word_lemmatized = lemmatizer.lemmatize(word, pos_tags_translation[pos_simple])
            lemmatized.append((word, word.lower(), word_lemmatized, word_lemmatized.lower()))
        else:
            word_lemmatized = lemmatizer.lemmatize(word)
            lemmatized.append((word, word.lower(), word_lemmatized, word_lemmatized.lower()))

    return lemmatized

def deduplicate_tuples(lemmatized_tokens):
    return tuple(sorted(set(lemmatized_tokens)))
