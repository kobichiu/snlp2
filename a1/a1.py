from gensim.models import KeyedVectors
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd

nlp = spacy.load("en_core_web_sm")
nlp.disable_pipes("parser", "ner")
print('Loading spaCy...')

file = "GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(file, binary=True, limit=100000)
print('Loading word2vec embeddings...')
model.save_word2vec_format("word2vec_embeddings.bin")


def load_data(filename):
    processing_dict = {}
    try:
        processing = pd.read_csv(filename)
    except Exception as e:
        raise ValueError(f"Failed to read {filename}, as the error is {e}")
    if processing.empty:
        raise ValueError(f'Failed to read {filename}, it is empty')
    for id, row in processing.iterrows():
        processing_dict[id] = {'description': row['description'], 'url': row['url']}
    return processing_dict


def preprocess_text(text):
    doc = nlp(text)
    preprocessed_tokens = []
    for token in doc:
        if not(token.is_punct or token.like_url or token.is_space or token.is_stop):
            qualified_token = token.lower_
            preprocessed_tokens.append(qualified_token)
    return preprocessed_tokens


def preprocess_texts(data_dict):
    for id in data_dict:
        to_be_processed = data_dict[id]['description']
        lists = preprocess_text(to_be_processed)
        data_dict[id]['pp_text'] = lists
    return data_dict


def get_vector(tokens):
    vectors = []
    for token in tokens:
        if token in model:
            vectors.append(model[token])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None


def get_vectors(data_dict):
    for id in data_dict:
        to_be_processed = data_dict[id]['pp_text']
        vector_list = get_vector(to_be_processed)
        data_dict[id]['vector'] = vector_list
    return data_dict


def cosine_similarity(v1, v2):
    dot_prod = dot(v1, v2)
    v1_denom = norm(np.sqrt(np.sum(np.power(v1, 2))))
    v2_denom = norm(np.sqrt(np.sum(np.power(v2, 2))))
    cos_sim = dot_prod / (v1_denom * v2_denom)
    return cos_sim


def k_most_similar(query, data_dict, k=5):
    query_tokens = preprocess_text(query)
    query_vector = get_vector(query_tokens)
    if query_tokens is None or query_vector is None:
        return []
    similarity_talks = []
    for id, nested_dict in data_dict.items():
        if nested_dict.get('vector') is not None:
            cos_sim = cosine_similarity(query_vector, nested_dict['vector'])
            similarity_talks.append((id, cos_sim))
    similarity_talks.sort(key=lambda x: x[1], reverse=True)
    return similarity_talks[:k]


def recommender_app(data_dict):
    while True:
        prompt = f"\nWhat is your query? "
        query = input(prompt)
        if query == "quit":
            return
        else:
            search_result = k_most_similar(query, data_dict)
        if not search_result:
            print("No results found.")
        else:
            print("\nTop 5 most similar talks to your query:")
            for id, sim in search_result:
                print(f"Description: {data_dict[id]['description']}")
                print(f"URL: {data_dict[id]['url']}\n")
            print(search_result)


def main():
    filename = 'ted_main.csv'
    data_dict = load_data(filename)
    data_dict = preprocess_texts(data_dict)
    data_dict = get_vectors(data_dict)
    recommender_app(data_dict)


if __name__ == '__main__':
    main()