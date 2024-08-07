# Clustering Evaluation for Crisis Management

This repository contains the experimental code of the clustering evaluation described in the paper **"Information Overload in Crisis Management: Bilingual Evaluation of Embedding Models for Clustering Social Media Posts in Emergencies"** by Markus Bayer, Marc André Kaufhold, and Christian Reuter.

## Overview

The script `main.py` is the primary script used to create embeddings, perform clustering, and evaluate the results based on different embedding models. The goal is to evaluate the effectiveness of various embedding models for clustering social media posts during emergencies, considering both English and German posts.

## Usage

### Parameters

The following parameters can be adjusted to customize the execution of the script:

- **models_types_to_use**: List of embedding models to use. Options include:
  - "word2vec_twitter"
  - "word2vec_crisis_1"
  - "word2vec_crisis_2"
  - "glove"
  - "fasttext_english"
  - "fasttext_german"
  - "sif"
  - "infersent_glove"
  - "infersent_fasttext"
  - "sent2vec_unigrams"
  - "sent2vec_bigrams"
  - (Discontinued Models)
    - "use_base"
    - "use_large"
    - "nli_mean_base"
    - "nli_mean_large"
    - "nli_mean_sts_base"
    - "nli_mean_sts_large"

- **models_to_use**: Dictionary specifying the SBERT and USE models to be used. The logic needs to be reimplemented if new models are to be included as the used models are discontinued.

- **CLUSTER_ALGORITHM**: The clustering algorithm to use. Options include:
  - "KMeans"
  - "SphericalKMeans"
  - "AgglomerativeClustering"

- **NORMALIZE**: Boolean flag to normalize data.
- **TRANSLATE**: Boolean flag to translate data.
- **REMOVE_STOPWORDS**: Boolean flag to remove stopwords.
- **EXTENDED_DATA**: Boolean flag to use extended data.
- **APPEND_OTHER_FEATURES**: Boolean flag to append other features.

### Dataset

The dataset used for the evaluation can be modified by changing the dataset path at line 95 in the `main.py` script.

### Running the Script

To run the script with the default parameters, execute the following command:

```
python main.py
```

To customize the parameters, you can modify them directly in the script or create a configuration file and parse it within the script.

## Contribution

Contributions are welcome! If you have any suggestions, find a bug, or want to improve the code, please open an issue or submit a pull request.


## Contact

For any questions or inquiries, please contact the authors of the paper:

- Markus Bayer
- Marc André Kaufhold
- Christian Reuter
