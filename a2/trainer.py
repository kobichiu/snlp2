import torch
import torch.nn as nn
import copy
import json
from sklearn.metrics import f1_score
from constants import *


class FeedForwardNet(nn.Module):
    def __init__(self, n_dims, hidden_size, n_classes):
        super(FeedForwardNet, self).__init__()
        self.linear1 = nn.Linear(n_dims, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

class Trainer:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_dev = None
        self.y_dev = None
        self.label_map = None
        self.n_dims = None
        self.n_classes = None
        self.best_model = None

    def _load_train_tensors(self, train_tensor_file):
        train_to_load = torch.load(train_tensor_file)
        self.X_train = train_to_load[X_KEY]
        self.y_train = train_to_load[Y_KEY]
        self.label_map = train_to_load[MAP_KEY]

    def _load_dev_tensors(self, dev_tensor_file):
        dev_to_load = torch.load(dev_tensor_file)
        self.X_dev = dev_to_load[X_KEY]
        self.y_dev = dev_to_load[Y_KEY]
        self.label_map = dev_to_load[MAP_KEY]

    def load_data(self, train_tensor_file, dev_tensor_file):
        self._load_train_tensors(train_tensor_file)
        self._load_dev_tensors(dev_tensor_file)
        self.n_dims = self.X_train.size(dim=1)
        self.n_classes = len(self.label_map)

    def _macro_f1(self, model):
        with torch.no_grad():
            model.eval()
            dev_prediction = model.forward(self.X_dev)
            dev_predicted_classes = torch.argmax(dev_prediction, dim=1)
            macro_f1 = f1_score(self.y_dev, dev_predicted_classes, average='macro')
            return macro_f1

    def _training_loop(self, model, loss_fn, optimizer, n_epochs):
        best_f1 = float(0)
        best_epoch = 0
        best_model = None

        for epoch in range(n_epochs):
            predictions = model.forward(self.X_train)
            loss = loss_fn(predictions, self.y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            macro_f1 = self._macro_f1(model)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_epoch = epoch + 1
                best_model = copy.deepcopy(model.state_dict())

        returned_dict = {MODEL_STATE_KEY: best_model,
                         F1_MACRO_KEY: best_f1,
                         BEST_EPOCH_KEY: best_epoch}

        return returned_dict

    def train(self, hidden_size, n_epochs, learning_rate):
        torch.manual_seed(42)
        model = FeedForwardNet(self.n_dims, hidden_size, self.n_classes)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        outcome_dict = self._training_loop(model, loss_fn, optimizer, n_epochs)

        outcome_dict.update({HIDDEN_SIZE_KEY: hidden_size,
                             N_DIMS_KEY: self.n_dims,
                             N_CLASSES_KEY: self.n_classes,
                             LEARNING_RATE_KEY: learning_rate,
                             N_EPOCHS_KEY: n_epochs,
                             OPTIMIZER_NAME_KEY: optimizer.__class__.__name__,
                             LOSS_FN_NAME_KEY: loss_fn.__class__.__name__})

        self.best_model = outcome_dict

        return outcome_dict

    def save_best_model(self, base_filename):
        pt_dict_keys = [MODEL_STATE_KEY, N_DIMS_KEY, N_CLASSES_KEY, HIDDEN_SIZE_KEY]
        pt_dict = {i: self.best_model[i] for i in pt_dict_keys}

        json_dict = self.best_model
        json_dict.pop(MODEL_STATE_KEY)

        torch.save(pt_dict, base_filename+".pt")
        with open(base_filename+"-info.json", 'w') as json_file:
            json.dump(json_dict, json_file, indent=4)

if __name__ == '__main__':
    text_trainer = Trainer()
    text_trainer.load_data("Data/train_tensor.pt", "Data/dev_tensor.pt")
    dictionary = text_trainer.train(8, 200, .01)
    text_trainer.save_best_model("Data/train_model")