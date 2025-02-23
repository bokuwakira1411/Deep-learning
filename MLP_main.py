from conda_build.cli.validators import validate_is_conda_pkg_or_recipe_dir

import MLP_setup
import MLP_train
import torch
def main():
    hidden_nodes = [64,32]
    epoch = 100
    learning_rates = [.00001, 0.0001, 0.001, 0.01, 0.1]
    (train_loader, validation_loader) = MLP_train.load_data()
    model = MLP_setup.Net(hidden_nodes, "sigmoid")
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rates[0])
    loss_vector = []
    acc_vector = []
    for e in range(1, epoch+1):
        MLP_train.train(e, model, train_loader, optimizer)
        MLP_train.validate(loss_vector, acc_vector, model, validation_loader)
main()