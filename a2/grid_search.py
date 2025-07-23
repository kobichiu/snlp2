from trainer import Trainer
from constants import *

def main():
    epochs = [100, 200, 300]
    learning_rates = [0.005, 0.01, 0.05]
    hidden_sizes = [68, 128, 256]
    best_f1 = 0.0

    for h_size in hidden_sizes:
        for n_epochs in epochs:
            for lr in learning_rates:
                tmp_model = Trainer()
                tmp_model.load_data("Data/train_tensor.pt", "Data/dev_tensor.pt")
                training = tmp_model.train(h_size, n_epochs, lr)
                tmp_model_score = training[F1_MACRO_KEY]

                if tmp_model_score > best_f1:
                    best_f1 = tmp_model_score
                    best_hypar = {
                        BEST_EPOCH_KEY: n_epochs,
                        LEARNING_RATE_KEY: lr,
                        HIDDEN_SIZE_KEY: h_size
                    }

    retrain_best_model = Trainer()
    retrain_best_model.load_data("Data/train_tensor.pt", "Data/dev_tensor.pt")
    retraining = retrain_best_model.train(best_hypar[HIDDEN_SIZE_KEY],
                                          best_hypar[BEST_EPOCH_KEY],
                                          best_hypar[LEARNING_RATE_KEY])
    print(retraining)
    retrain_best_model.save_best_model("Models/best-model")


if __name__ == '__main__':
    main()
