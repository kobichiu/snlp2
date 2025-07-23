import torch
from sklearn.metrics import classification_report
from trainer import FeedForwardNet
from constants import *


class Evaluator:
    def __init__(self):
        self.model = None
        self.X_test = None
        self.y_test = None
        self.label_map = None

    def load_model(self, model_file):
        loader = torch.load(model_file)
        n_dims, hidden_size, n_classes = loader[N_DIMS_KEY], loader[HIDDEN_SIZE_KEY], loader[N_CLASSES_KEY]
        tbc_model = FeedForwardNet(n_dims, hidden_size, n_classes)
        tbc_model.load_state_dict(loader[MODEL_STATE_KEY])
        tbc_model.eval()
        self.model = tbc_model

    def load_data(self, data_file):
        loading = torch.load(data_file)
        self.X_test = loading[X_KEY]
        self.y_test = loading[Y_KEY]
        self.label_map = loading[MAP_KEY]

    def evaluate_model(self):
        with torch.no_grad():
            label_map = list(self.label_map.keys())
            test_prediction = self.model.forward(self.X_test)
            test_predicted_classes = torch.argmax(test_prediction, dim=1)
            report_dict = classification_report(self.y_test, test_predicted_classes,
                                                target_names=label_map, output_dict=True)
            report_str = classification_report(self.y_test, test_predicted_classes,
                                               target_names=label_map, output_dict=False)
        return report_dict, report_str


if __name__ == '__main__':
    evaluation0 = Evaluator()
    Evaluator.load_model(evaluation0, "Models/baseline-model-given.pt")
    Evaluator.load_data(evaluation0, "Data/dev_tensor.pt")
    dict0, string0 = Evaluator.evaluate_model(evaluation0)
    print("Evaluation of the baseline model on the dev data:")
    print(dict0)
    print(string0)

    evaluation1 = Evaluator()
    Evaluator.load_model(evaluation1, "Models/baseline-model-given.pt")
    Evaluator.load_data(evaluation1, "Data/test_tensor.pt")
    dict1, string1 = Evaluator.evaluate_model(evaluation1)
    print("Evaluation of the baseline model on the test data:")
    print(dict1)
    print(string1)

    evaluation2 = Evaluator()
    Evaluator.load_model(evaluation2, "Models/best-model.pt")
    Evaluator.load_data(evaluation2, "Data/dev_tensor.pt")
    dict2, string2 = Evaluator.evaluate_model(evaluation2)
    print("Evaluation of the my best model on the dev data:")
    print(dict2)
    print(string2)

    evaluation3 = Evaluator()
    Evaluator.load_model(evaluation3, "Models/best-model.pt")
    Evaluator.load_data(evaluation3, "Data/test_tensor.pt")
    dict3, string3 = Evaluator.evaluate_model(evaluation3)
    print("Evaluation of the baseline model on the test data:")
    print(dict3)
    print(string3)