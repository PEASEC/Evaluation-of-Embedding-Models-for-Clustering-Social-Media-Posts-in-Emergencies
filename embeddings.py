import re
import numpy as np
import os
import io
import gensim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.fasttext import load_facebook_model
import nltk
from nltk.corpus import stopwords
from nltk import TweetTokenizer
import mxnet as mx
from bert_embedding import BertEmbedding
import data_io, params, SIF_embedding
from models import InferSent
import torch
import sent2vec
import word2vecReaderUtils
import aidrtokenize
import fasttext


nltk.download("punkt")
nltk.download("stopwords")
dirname = os.path.dirname(__file__)


def tokenize_tweet_glove(tweet):
    tweet = tweet.lower()
    tweet = (
        tweet.replace('"', "")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .replace("  ", " ")
        .replace("  ", " ")
    )

    if tweet.startswith("rt "):
        tweet = "<retweet> " + tweet[3:]

    tweet = re.sub("\b(\S*?)(.)\2{2,}\b")

    tweet = re.sub(
        "(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]",
        "<url>",
        tweet,
    )

    tweet = re.sub("[0-9]+", "<number>", tweet)

    tweet = re.sub("#", " <hashtag> ", tweet)

    tweet = re.sub(
        "(?<=^|(?<=[^a-zA-Z0-9-\\.]))@[A-Za-z0-9-]+(?=[^a-zA-Z0-9-_])", "<user>", tweet
    )

    tweet = (
        tweet.replace("   ", " ")
        .replace("  ", " ")
        .replace(":)", "<smile>")
        .replace("<3", "<heart>")
        .replace(":(", "<sadface>")
    )

    tweet = (
        tweet.replace(".", " . ")
        .replace(",", " , ")
        .replace(":", " : ")
        .replace("…", "")
        .replace(";", " ; ")
        .replace("!", " ! ")
        .replace("?", " ? ")
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace("-", " - ")
        .replace("[", " [ ")
        .replace("]", " ] ")
        .replace('"', ' " ')
        .replace("{", " { ")
        .replace("}", " } ")
        .replace("'", " ' ")
        .replace("=", " = ")
        .replace("   ", " ")
        .replace("  ", " ")
    )

    return tweet.split()


def tokenize_tweet_fasttext(tweet):
    # https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc#L147

    return (
        tweet.replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .replace("\v", " ")
        .replace("\f", " ")
        .replace("\0", " ")
        .split(" ")
    )


def tokenize_tweet_crisis(tweet):
    return aidrtokenize.tokenizeRawTweetText(tweet)


def sent2vec_tokenize_tweets(tweets):
    def sent2vec_preprocess_tweet(tweet):
        tweet = tweet.lower()
        tweet = re.sub(
            "((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))", "<url>", tweet
        )
        tweet = re.sub("(\@[^\s]+)", "<user>", tweet)
        try:
            tweet = tweet.decode("unicode_escape").encode("ascii", "ignore")
        except:
            pass
        return tweet

    tknzr = TweetTokenizer()
    output = []
    for tweet in tweets:
        tweet = tknzr.tokenize(tweet)
        output.append(sent2vec_preprocess_tweet(" ".join(tweet)))

    return output


def get_token_embeddings(document, word_vectors, is_glove, fastText=False):
    embedding_document = []
    unknown_count = 0
    known_count = 0

    for token in document:
        if fastText:
            known_count += 1
            embedding_document.append(word_vectors.get_word_vector(token))
        elif token in word_vectors:
            known_count += 1
            embedding_document.append(word_vectors[token])
        else:
            unknown_count += 1
            # glove has an extra unknown vector
            if is_glove:
                embedding_document.append(word_vectors["<unknown>"])

    return np.array(embedding_document), unknown_count, known_count


def get_document_embedding(document_embeddings, type, embedding_dim):
    # if whole document is empty generate a random vector
    if len(document_embeddings) == 0:
        return np.random.randn(embedding_dim)

    if type == "Average":
        return np.average(document_embeddings, axis=0)
    if type == "MinMax":
        return np.concatenate(
            (np.max(document_embeddings, axis=0), np.min(document_embeddings, axis=0))
        )


def get_word2vec_document_embeddings(
    documents,
    remove_stopwords,
    document_embedding_type,
    embedding_path,
    binary,
    is_glove,
):
    document_embeddings = []
    unknown_count = 0
    known_count = 0

    # init embeddings
    if is_glove:
        glove2word2vec(
            glove_input_file=os.path.join(dirname, embedding_path),
            word2vec_output_file=os.path.join(dirname, "gensim_vectors.txt"),
        )
        embedding_path = "gensim_vectors.txt"
        binary = False

    kv = KeyedVectors.load_word2vec_format(
        os.path.join(dirname, embedding_path), binary=binary, unicode_errors="ignore"
    )

    embedding_dim = kv.vector_size
    word_vectors = kv.wv

    for document in documents:

        if "crisis" in embedding_path:
            document = tokenize_tweet_crisis(document)
        elif "word2vec_twitter" == embedding_path:
            document = word2vecReaderUtils.simple_preprocess(document)
        else:
            document = tokenize_tweet_glove(document)

        if remove_stopwords:
            stop_words = set(stopwords.words("english"))
            document = [token for token in document if not token in stop_words]

        token_embeddings, unknown_count_tmp, known_count_tmp = get_token_embeddings(
            document, word_vectors, is_glove
        )

        unknown_count += unknown_count_tmp
        known_count += known_count_tmp

        document_embeddings.append(
            get_document_embedding(
                token_embeddings, document_embedding_type, embedding_dim
            )
        )

    return np.array(document_embeddings), unknown_count, known_count


def get_fasttext_document_embeddings(
    documents, remove_stopwords, document_embedding_type, embedding_path
):

    def load_fasttext_vectors(embedding_path):
        fin = io.open(
            embedding_path, "r", encoding="utf-8", newline="\n", errors="ignore"
        )
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(" ")
            data[tokens[0]] = map(float, tokens[1:])
        fin.close()
        return data

    document_embeddings = []
    unknown_count = 0
    known_count = 0

    # init embeddings
    word_vectors = fasttext.load_model(embedding_path)

    for document in documents:
        document = tokenize_tweet_fasttext(document)

        if remove_stopwords:
            stop_words = set(stopwords.words("english"))
            document = [token for token in document if not token in stop_words]

        token_embeddings, unknown_count_tmp, known_count_tmp = get_token_embeddings(
            document, word_vectors, False, True
        )
        unknown_count += unknown_count_tmp
        known_count += known_count_tmp

        document_embeddings.append(
            get_document_embedding(token_embeddings, document_embedding_type, 0)
        )

    return np.array(document_embeddings), unknown_count, known_count


# https://github.com/PrincetonML/SIF/blob/master/examples/sif_embedding.py
def get_sif_document_embeddings(documents, embedding_path, weight_path):
    weightpara = 1e-3
    rmpc = 1

    # load word vectors
    (words, We) = data_io.getWordmap(embedding_path)

    # load word weights
    word2weight = data_io.getWordWeight(weight_path, weightpara)
    weight4ind = data_io.getWeight(words, word2weight)

    # load sentences
    x, m, _ = data_io.sentences2idx(
        documents, words
    )  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind)  # get word weights

    # set parameters
    params = params.params()
    params.rmpc = rmpc
    # get SIF embedding
    embedding = SIF_embedding.SIF_embedding(
        We, x, w, params
    )  # embedding[i,:] is the embedding for sentence i

    return embedding, None, None


def get_use_document_embeddings(documents, model):
    embeddings = model(documents)

    return embeddings, None, None


def get_infersent_document_embeddings(documents, embedding_path, pretrained_path):
    V = 2
    params_model = {
        "bsize": 64,
        "word_emb_dim": 300,
        "enc_lstm_dim": 2048,
        "pool_type": "max",
        "dpout_model": 0.0,
        "version": V,
    }

    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(pretrained_path))
    infersent.set_w2v_path(embedding_path)

    infersent.build_vocab(documents, tokenize=True)
    embeddings = infersent.encode(documents, tokenize=True)

    return embeddings, None, None


def get_sent2vec_document_embeddings(documents, model_path):
    # sent2vec tokenization
    documents = sent2vec_tokenize_tweets(documents)

    model = sent2vec.Sent2vecModel()
    model.load_model(model_path)
    embeddings = model.embed_sentences(documents)

    return embeddings, None, None


def get_sbert_document_embeddings(documents, model):
    embeddings = model.encode(documents)
    return embeddings, None, None


def get_bert_word_document_embeddings(documents, document_embedding_type):
    document_embeddings = []
    unknown_count = 0
    known_count = 0

    bert_embedding = BertEmbedding(
        model="bert_12_768_12", dataset_name="wiki_multilingual_cased"
    )

    for document in documents:
        embeddings = bert_embedding(document)

        tmp_embeddings = []
        for sentence in embeddings:
            tmp_embeddings.extend(sentence[1])
            # for vector in sentence[1]:
            #    tmp_embeddings.append(vector)

        document_embeddings.append(
            get_document_embedding(np.array(tmp_embeddings), document_embedding_type)
        )

    return document_embeddings, unknown_count, known_count


def get_document_embeddings(
    documents, model_type, remove_stopwords, document_embedding_type, model=None
):
    if model_type == "word2vec_twitter":
        # Tokenizer?
        embedding_path = "word2vec_twitter_model.bin"
        return get_word2vec_document_embeddings(
            documents,
            remove_stopwords,
            document_embedding_type,
            embedding_path,
            True,
            False,
        )
    elif model_type == "word2vec_crisis_1":
        embedding_path = "crisisNLP_word_vector.bin"
        return get_word2vec_document_embeddings(
            documents,
            remove_stopwords,
            document_embedding_type,
            embedding_path,
            True,
            False,
        )
    elif model_type == "word2vec_crisis_2":
        embedding_path = "crisis_word_vector.txt"
        return get_word2vec_document_embeddings(
            documents,
            remove_stopwords,
            document_embedding_type,
            embedding_path,
            False,
            False,
        )
    elif model_type == "glove":
        embedding_path = "glove.twitter.27B.300d.txt"
        return get_word2vec_document_embeddings(
            documents,
            remove_stopwords,
            document_embedding_type,
            embedding_path,
            False,
            True,
        )
    elif model_type == "fasttext_english":
        embedding_path = "wiki.en.bin"
        return get_fasttext_document_embeddings(
            documents, remove_stopwords, document_embedding_type, embedding_path
        )
    elif model_type == "fasttext_german":
        embedding_path = "wiki.de.bin"
        return get_fasttext_document_embeddings(
            documents, remove_stopwords, document_embedding_type, embedding_path
        )
    elif model_type == "sif":
        embedding_path = "glove.840B.300d.txt"
        weight_path = "enwiki_vocab_min200.txt"
        return get_sif_document_embeddings(documents, embedding_path, weight_path)
    elif model_type == "use_base" or model_type == "use_large":
        return get_use_document_embeddings(documents, model)
    elif model_type == "infersent_glove":
        embedding_path = "glove.840B.300d.txt"
        pretrained_path = "infersent1.pkl"
        return get_infersent_document_embeddings(
            documents, embedding_path, pretrained_path
        )
    elif model_type == "infersent_fasttext":
        embedding_path = "crawl-300d-2M.vec"
        pretrained_path = "infersent2.pkl"
        return get_infersent_document_embeddings(
            documents, embedding_path, pretrained_path
        )
    elif model_type == "sent2vec_unigrams":
        model_path = "twitter_unigrams.bin"
        return get_sent2vec_document_embeddings(documents, model_path)
    elif model_type == "sent2vec_bigrams":
        model_path = "twitter_bigrams.bin"
        return get_sent2vec_document_embeddings(documents, model_path)
    elif (
        model_type == "nli_mean_base"
        or model_type == "nli_mean_large"
        or model_type == "nli_mean_sts_base"
        or model_type == "nli_mean_sts_large"
    ):
        return get_sbert_document_embeddings(documents, model)
