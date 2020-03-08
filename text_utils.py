import re
from nltk.corpus import stopwords
from constants import *


def remove_non_ascii(text):
    """
    Remove non ascii characters in the text

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(r'[^\x00-\x7F]', ' ', text)


def remove_punctuation(text):
    """
    Remove all punctuation in the text

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(r'[^\w]', ' ', text)


def remove_digits(text):
    """
    Remove all digits in the text

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(r'[\d]', '', text)


def to_lowercase(text):
    """
    Lowercase the text

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return text.lower()


def remove_extra_space(text):
    """
    Remove extra space between two words

    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(' +', ' ', text)


def remove_url(text):
    """
    Remove urls in the string
    Parameters
    ----------
    text : str

    Returns
    -------
    text: str
    """
    return re.sub(r'http\S+', ' ', text)


def remove_underline(text):
    return text.replace('_', ' ')


def remove_less_frequent_words(words, less_freq):
    """
    Remove less frequent words in the text
    Parameters
    ----------
    words : word list
    less_freq: less frequent word
    Returns
    -------
    words: list
    """
    less_freq = dict((k, 1) for k in less_freq)
    words = [w for w in words if w not in less_freq]

    return words


def remove_stopwords(words):
    """
    Remove stop words in the text
    Parameters
    ----------
    words : list

    Returns
    -------
    words: list
    """
    stop_words = set(stopwords.words('english'))

    # read the stop words
    stop_words_extra = []
    with open(STOP_WORDS) as f:
        stop_words_extra = [line.rstrip() for line in f]

    words = [w for w in words if w not in stop_words and w not in stop_words_extra]

    return words
