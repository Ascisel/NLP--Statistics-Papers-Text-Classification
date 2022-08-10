import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
import string


def add_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse text columns. 

    Parameters:
    -----------
    df - dataset with 3 columns

    Returns:
    --------
    df - dataset with parsed columns
    """
    df["text"] = (
        df["titles"].astype(str)
        + ". "
        + df["abstracts"].astype(str)
    )  #combain all text columns into one
    df["text_sents"] = df["text"].apply(lambda text: __tokenize_sents(text))

    df["sents_alpha"] = df["text_sents"].apply(lambda sents: __get_alpha_sents(sents))
    df["words_alpha"] = df["sents_alpha"].apply(lambda alpha: __get_alpha_words(alpha))

    df["words_stem"] = df["words_alpha"].apply(lambda words: __get_stem_words(words))
    df["words_stem"] = df["words_stem"].map(__remove_punctuations)

    df["words_lemma"] = df["words_alpha"].apply(lambda words: __get_lemma_words(words))
    df["words_lemma"] = df["words_lemma"].map(__remove_punctuations)

    return df


# ALPHA [LETTERS ONLY]


def __parse_alpha(text: str) -> list:
    text = re.sub("[^a-zA-Z ]", "", text)
    words = __split_text(text)
    return words


def __get_alpha_sents(sents: list) -> list:
    parsed_sents = []
    for sent in sents:
        parsed_sent = __parse_alpha(sent)
        parsed_sents.append(parsed_sent)
    return parsed_sents


def __get_alpha_words(parsed_sents: list) -> list:
    words = __merge_sents(parsed_sents)
    return words

# STEM [STEMMING]


def __stem(words: list) -> list:
    porter = nltk.stem.PorterStemmer()
    stems = []
    for word in words:
        stems.append(porter.stem(word))
    return stems

def __get_stem_words(words: list) -> list:
    stems = __stem(words)
    return stems


# LEMMA [LEMMATIZATION]


def __lemmatize(words: list) -> list:
    wordnet_lematizer = nltk.stem.WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemmas.append(wordnet_lematizer.lemmatize(word))
    return lemmas

def __get_lemma_words(words: list) -> list:
    lemmas = __lemmatize(words)
    return lemmas


# BASIC FUNCTIONS


def __tokenize_sents(text: str):
    return nltk.tokenize.sent_tokenize(text)

def __split_text(text: str) -> list:
    words = text.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    return words

def __merge_sents(sents: list) -> list:
    text = []
    for sent in sents:
        for word in sent:
            text.append(word)
    return text

def __remove_punctuations(text: str) -> str:
  return text.translate(str.maketrans('', '', string.punctuation))