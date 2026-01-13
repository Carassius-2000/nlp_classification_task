import re
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def check_download():
    resources = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]
    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def text_preprocessing(text_series):
    check_download()
    text_without_punctuation = text_series.str.lower().str.replace(
        re.compile(r"[^\w\s]+"), " ", regex=True
    )
    stopwords_ru = set(stopwords.words("russian"))
    stopwords_en = set(stopwords.words("english"))
    stopwords_all = stopwords_ru | stopwords_en
    morph = MorphAnalyzer()
    preprocessed_text = text_without_punctuation.apply(
        lambda x: " ".join(
            normalized_word
            for item in word_tokenize(x, language="russian")
            if item.isalpha()
            and (normalized_word := morph.parse(item)[0].normal_form)
            not in stopwords_all
        )
    )
    return preprocessed_text
