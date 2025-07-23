import spacy
import compress_fasttext
import torch
import numpy as np
import json
from constants import *

nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
print('Loading spaCy...')

emb = compress_fasttext.models.CompressedFastTextKeyedVectors.load("Data/fasttext-de-mini")
print('Loading embeddings...')

class Preprocessor:
    def __init__(self, spacy_model, embeddings):
        self.nlp = spacy_model
        self.embeddings = embeddings

        self.X_texts = None
        self.y_texts = None
        self.X_tensor = None
        self.y_tensor = None
        self.label_map = None

    def _load_csv_data(self, data_file):
        self.X_texts, self.y_texts = [], []

        with open(data_file, mode='r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(';', 1)

                topic, text = parts
                self.X_texts.append(text)
                self.y_texts.append(topic)

    def _load_label_map(self, label_map_file):
        self.label_map = {}

        file = open(label_map_file)
        self.label_map = json.load(file)

    def load_data(self, data_file, label_map_file):
        self._load_csv_data(data_file)
        self._load_label_map(label_map_file)

    def _preprocess_text(self, text):
        result = []

        for token in self.nlp(text):
            if not (token.is_stop or token.is_punct or token.is_space or token.like_num):
                result.append(token.text)

        return result

    def _calc_mean_embedding(self, text):
        preprocessed_text = self._preprocess_text(text)
        embeddings = []

        for token in preprocessed_text:
            if token in self.embeddings:
                embeddings.append(self.embeddings[token])

        embeddings_np = np.array(embeddings)

        mean_embedding = torch.tensor(embeddings_np, dtype=torch.float32).mean(dim=0)
        return mean_embedding

    def _generate_X_tensor(self):
        emb_list = []

        for text in self.X_texts:
            mean_emb = self._calc_mean_embedding(text)
            emb_list.append(mean_emb)

        self.X_tensor = torch.stack(emb_list).to(dtype=torch.float32)

    def _generate_y_tensor(self):
        gold_class_codes = []

        for label in self.y_texts:
            gold_class_codes.append(self.label_map.get(label))

        self.y_tensor = torch.tensor(gold_class_codes)

    def generate_tensors(self):
        self._generate_X_tensor()
        self._generate_y_tensor()

    def save_tensors(self, tensor_file):
        tensor_dict = {
            X_KEY: self.X_tensor,
            Y_KEY: self.y_tensor,
            MAP_KEY: self.label_map
        }
        torch.save(tensor_dict, tensor_file)

if __name__ == '__main__':
    train = "Data/train-data.csv"
    dev = "Data/dev-data.csv"
    test = "Data/test-data.csv"
    label = "Data/label_map.json"

    train_tensor = "Data/train_tensor.pt"
    dev_tensor = "Data/dev_tensor.pt"
    test_tensor = "Data/test_tensor.pt"

    pre_processor = Preprocessor(nlp, emb)

    pre_processor.load_data(train, label)
    pre_processor.generate_tensors()
    pre_processor.save_tensors(train_tensor)
    print("Preprocessing training data done!")

    pre_processor.load_data(dev, label)
    pre_processor.generate_tensors()
    pre_processor.save_tensors(dev_tensor)
    print("Preprocessing dev data done!")

    pre_processor.load_data(test, label)
    pre_processor.generate_tensors()
    pre_processor.save_tensors(test_tensor)
    print("Preprocessing test data done!")