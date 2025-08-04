from typing import List, Set, Dict, Tuple, Union

import logging
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from . import env, utils


logging.basicConfig(level='INFO')


def tfidf(
        vocabulary: Set[str],
        sentences: List[str],
        n_components: int = None,
        return_features: bool = False,
) -> Union[Dict[int, np.array], Tuple[Dict[int, np.array], List[str]]]:
    """Compute tfidf embeddings for a list of strings

    Args:
        vocabulary: The vocabulary to use
        sentences: texts
        n_components: number of components of the dimensionality reduction if any
        return_features: whether to return features along with embeddings

    Returns:
        Just a map nodes and their embeddings, or that along with the features
    """

    logging.info('Computing TFIDF Embeddings...')
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    features = vectorizer.get_feature_names()
    X = vectorizer.fit_transform(sentences).toarray()

    if n_components:
        # Dimensionality reduction
        X = utils.truncated_svd(X, n_components=n_components)
        # Normalization
        normalizer = Normalizer(copy=False)
        X = normalizer.fit_transform(X)

    embeddings = X.tolist()

    if return_features:
        return embeddings, features
    return embeddings


def bert(sentences: List[str]) -> List[np.array]:
    """Compute bert embeddings for nodes of a knowledge graph

    Args:
        sentences: text to use for each node

    Returns:
        A map nodes and their embeddings
    """

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Compute embedding for both lists
    embeddings = model.encode(sentences, convert_to_tensor=True, batch_size=env.sbert.encode_batch_size)

    return [embed.cpu().numpy() for embed in embeddings]


def fasttext(texts: List[str]) -> List[np.array]:

    """Compute fasttext embeddings for texts.

    Args:
        texts: text to use for each node

    Returns:
        A map nodes and their embeddings
    """

    # initialize the word embeddings
    word_embedding = WordEmbeddings('en')

    # initialize the document embeddings, mode = mean
    document_embeddings = DocumentPoolEmbeddings([word_embedding])

    # sentences

    sentences = [Sentence(text) for text in texts]

    for sentence in sentences:
        document_embeddings.embed(sentence)

    # Compute embeddings
    embeddings = [sentence.embedding for sentence in sentences]

    return embeddings


#
# Display Embeddings
#
def plot(embeddings, colors=None, labels=None):
    """Plot nodes on a 2D plane using dimensionality reduction.

    Args:
        embeddings: A list of embeddings
        colors: A list of colors
        labels: labels for points
    """

    v2d = utils.truncated_svd(np.array(embeddings), n_components=2)
    x = v2d[:, 0]
    y = v2d[:, 1]
    plt.scatter(x, y,  c=colors)
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (x[i], y[i]))
    plt.show()
