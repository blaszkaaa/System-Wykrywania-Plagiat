import os
import string
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(file):
    with open(file, "r") as f:
        text = f.read()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokenized_words = text.split()
    final_text = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_text.append(word)
    return final_text


def cosine_similarity_text(list_of_files):
    documents = [open(f, encoding='utf-8').read() for f in list_of_files]
    vectorizer = CountVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)
    return csim


def check_plagiarism():
    path = os.getcwd()
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    matrix = cosine_similarity_text(files)

    df = pd.DataFrame(matrix, index=files, columns=files)
    print(df)


if __name__ == "__main__":
    check_plagiarism()
