from typing import List, Set, Dict, Tuple, Union

import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence


logging.basicConfig(level='INFO')


#
# Classical Embeddings
#
def tfidf(
        kg: KG,
        nodes: List[int],
        vocabulary: Set[str],
        weighted=False,
        radius=1,
        node_texts: List[str] = None,
        n_components: int = None,
        return_features: bool = False,
) -> Union[Dict[int, np.array], Tuple[Dict[int, np.array], List[str]]]:
    """Compute tfidf embeddings for nodes of a knowledge graph
    Args:
        kg: The knowledge graph
        nodes: The nodes (ids of the entities) for which we want embeddings
        vocabulary: The vocabulary to use
        weighted: whether to weight vocabulary words by their salience
        node_texts: text to use for each node
        radius: radius to get the node texts (ignored if node_texts is passed)
        n_components: number of components of the dimensionality reduction if any
        return_features: whether to return features along with embeddings
    Returns:
        Just a map nodes and their embeddings, or that along with the features
    """

    if node_texts:
        sentences = [node_texts[i] for i, node in enumerate(nodes)]
    else:
        logging.info('Getting text contexts...')
        sentences = [gs.text_context(kg, entity=kg.get_entity(node), radius=radius) for node in tqdm(nodes)]

    logging.info('Computing TFIDF Embeddings...')
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    features = vectorizer.get_feature_names()
    X = vectorizer.fit_transform(sentences).toarray()

    if weighted:
        logging.info('Weighting Embeddings...')
        for i, embedding in tqdm(list(enumerate(X))):
            for j, token in enumerate(features):
                weight = gs.get_salience(kg, nodes[i], term=token)
                embedding[j] *= weight

    if n_components:
        # Dimensionality reduction
        X = utils.truncated_svd(X, n_components=n_components)
        # Normalization
        normalizer = Normalizer(copy=False)
        X = normalizer.fit_transform(X)
    embeddings = dict(zip(nodes, X.tolist()))

    if return_features:
        return embeddings, features
    return embeddings


def bert(
    kg: KG,
    nodes: List[int],
    node_texts: List[str] = None,
    pool=False
) -> Dict[int, np.array]:
    """Compute bert embeddings for nodes of a knowledge graph
    Args:
        kg: The knowledge graph
        nodes: The nodes (ids of the entities) for which we want embeddings
        pool: whether to pool sentences embeddings to compute paragraph embeddings instead of just truncating
        node_texts: text to use for each node
    Returns:
        A map nodes and their embeddings
    """

    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    if pool:
        # Document
        documents = [kg.get_entity(node).text for node in nodes]
        document_sentences_offsets = []
        document_sentences_lengths = []
        all_sentences = []
        for document in documents:
            document_sentences_offsets.append(len(all_sentences))
            document_sentences = utils.doc2sentences(
                document)  # [sent for sent in utils.doc2sentences(document) if sent]
            document_sentences_lengths.append(len(document_sentences))
            all_sentences.extend(document_sentences)

        # Compute embeddings
        all_sentence_embeddings = model.encode(all_sentences, batch_size=env.sbert.encode_batch_size)

        embeddings = [
            torch.mean(
                torch.stack(
                    tuple(
                        torch.from_numpy(se)
                        for se in all_sentence_embeddings[
                                  document_sentences_offsets[i]:
                                  document_sentences_offsets[i] + document_sentences_lengths[i]
                                  ]
                    )
                ),
                dim=0,
            ) if document_sentences_lengths[i] > 0
            else torch.from_numpy(model.encode([''], batch_size=env.sbert.encode_batch_size)[0])
            for i in tqdm(range(len(documents)))
        ]
    else:
        # sentences
        if node_texts:
            sentences = [node_texts[i] for i, node in enumerate(nodes)]
        else:
            sentences = [kg.get_entity(node).text for node in kg.nx]
        # Compute embedding for both lists
        embeddings = model.encode(sentences, convert_to_tensor=True, batch_size=env.sbert.encode_batch_size)

    return {node: embeddings[i].cpu().numpy() for i, node in enumerate(nodes)}


def fasttext(
    kg: KG,
    nodes: List[int],
    node_texts: List[str] = None,
    radius=1,
    max_words=None
) -> Dict[int, np.array]:
    """Compute fasttext embeddings for nodes of a knowledge graph.
    Args:
        kg: The knowledge graph
        nodes: The nodes (ids of the entities) for which we want embeddings
        node_texts: text to use for each node
        radius: radius to get the node texts (ignored if node_texts is passed)
        max_words: truncation parameter for nodes if needed, generally not
    Returns:
        A map nodes and their embeddings
    """

    # initialize the word embeddings
    word_embedding = WordEmbeddings('fr')

    # initialize the document embeddings, mode = mean
    document_embeddings = DocumentPoolEmbeddings([word_embedding])

    # sentences

    if node_texts:
        sentences = [Sentence(node_texts[i][:max_words]) for i, node in enumerate(nodes)]
    else:
        sentences = [
            Sentence(gs.text_context(kg, entity=kg.get_entity(node), radius=radius)[:max_words])
            for node in tqdm(nodes)
        ]

    for sentence in sentences:
        try:
            document_embeddings.embed(sentence)
        except:
            pass

    # Compute embedding for both lists
    embeddings = [sentence.embedding for sentence in sentences]

    return {node: embeddings[i].cpu().numpy() for i, node in enumerate(nodes)}


#
# Display Embeddings
#
def plot_nodes(node_embeddings, node_colors=None, with_labels=False):
    """Plot nodes on a 2D plane using dimensionality reduction.
    Args:
        node_embeddings: A dict of nodes to plot, with their embeddings
        node_colors: A dict of nodes to plot, with their colors
        with_labels: whether to plot nodes with their labels (entity id)
    """

    v2d = utils.truncated_svd(np.array(list(node_embeddings.values())), n_components=2)
    x = v2d[:, 0]
    y = v2d[:, 1]
    colors = list(node_colors.values()) if node_colors else None
    plt.scatter(x, y,  c=colors)
    if with_labels:
        for i, node in enumerate(node_embeddings):
            plt.annotate(node, (x[i], y[i]))

    plt.show()