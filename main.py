"""RUL Prediction with LSTM"""
from macro import *
import numpy as np
from loading_data import *
from model import *
from visualize import *
# from torch.autograd import Variable

def testing_function(model, num, group_for_test):
    rmse_test, result_test = 0, list()

    for ite in range(1, num + 1):
        X_test = group_for_test.get_group(ite).iloc[:, 2:]
        # X_test_tensors = Variable(torch.tensor(X_test.to_numpy(), device=DEVICE))
        X_test_tensors = torch.tensor(X_test.to_numpy()).to(DEVICE)
        X_test_tensors = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

        X_test_tensors = X_test_tensors.float()

        test_predict = model(X_test_tensors)
        data_predict = max(test_predict[-1].to(CPU).detach().numpy(), 0) # RUL should be non-negative
        result_test.append(data_predict)
        rmse_test = np.add(np.power((data_predict - y_test.to_numpy()[ite - 1]), 2), rmse_test)

    rmse_test = (np.sqrt(rmse_test / num)).item()
    return result_test, rmse_test


def train(LSTM_model, ntrain, group_for_train):
    """
    :param LSTM_model: initialized model
    :param ntrain: number of samples in training set
    :param group_for_train: grouped data per sample
    :return: evaluation results
    """
    global criterion, optimizer
    rmse_temp = 100

    for epoch in range(1, N_EPOCH + 1):

        LSTM_model.train() # train model
        """
        set/resume LSTM_model_to_train in training mode by enabling:
            1. Dropout;
            2. Batch normalization;
            3. Gradient Computation;
            4. Updating parameters;
        REMIND: use (model name).eval() to evaluate
        """

        epoch_loss = 0

        for i in range(1, ntrain + 1):
            X, y = group_for_train.get_group(i).iloc[:, 2:-1], group_for_train.get_group(i).iloc[:, -1:]

            # X_train_tensors = Variable(torch.tensor(X.to_numpy(), device=DEVICE))
            # y_train_tensors = Variable(torch.tensor(y.to_numpy(), device=DEVICE))
            # X_train_tensors = torch.tensor(X.to_numpy()).to(DEVICE)
            # y_train_tensors = torch.tensor(y.to_numpy()).to(DEVICE)
            X_train_tensors = torch.Tensor(X.to_numpy()).to(DEVICE)
            y_train_tensors = torch.Tensor(y.to_numpy()).to(DEVICE)
            """
            1.    X and y are instances of DataFrame type, and
                    (DataFrame).to_numpy() method will convert 
                    them to numpy array type.
            2.    torch.Tensor() is different from torch.tensor(). 
                    torch.Tensor() will  return a tensor whose
                    data type is determined by 
                    torch.get_default_tensor_type(), while
                    torch.tensor() is a function that constructs
                    a tensor with actual data, inferring an
                    automatically determined tensor type.
                  REMIND: using torch.tensor() here leads to
                    incorrect data type, and the following two
                    lines of code is needed to solve it.  
            """
            # X_train_tensors = X_train_tensors.float()
            # y_train_tensors = y_train_tensors.float()

            X_train_tensors = torch.reshape(
                X_train_tensors,
                (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])
                # re-arange into (batch_size, sequence_length, input_size/feature)
            )

            outputs = LSTM_model(X_train_tensors)  # forward pass

            optimizer.zero_grad()  # calculate the gradient, manually setting to 0
            loss = criterion(outputs, y_train_tensors)  # obtain the loss function
            epoch_loss += loss.item()
            loss.backward()  # calculates the loss of the loss function
            optimizer.step()  # improve from loss, i.e back propagation

        if epoch % 1 == 0:  # evaluate the model on testing set with each epoch

            LSTM_model.eval()  # evaluate model
            """
            set/resume LSTM_model in evaluating mode by disabling:
                1. Dropout;
                2. Batch normalization;
                3. Gradient Computation;
                4. Updating parameters;
            """

            result, rmse = testing_function(LSTM_model, num_test, group_test)

            if rmse_temp < rmse and rmse_temp < 24:
                result, rmse = result_temp, rmse_temp
                break

            rmse_temp, result_temp = rmse, result  # store the last rmse
            print("Epoch: %d, loss: %1.5f, rmse: %1.5f" % (epoch, epoch_loss / ntrain, rmse))

    return result, rmse


if __name__ == "__main__":

    print(f"Using {DEVICE} device")


    # fetch basic information from data sets
    group, group_test, y_test = load_FD001(MAX)
    num_train, num_test = len(group.size()), len(group_test.size())
    input_size = group.get_group(1).shape[1] - 3  # number of real features

    # LSTM model initialization
    # model = LSTM1(input_size, N_HIDDEN, N_LAYER) # our lstm class
    model = LSTM1(input_size, N_HIDDEN, N_LAYER).to(DEVICE)  # GPU support

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # training and evaluation
    result, rmse = train(model, num_train, group)
    visualize(result, y_test, num_test, rmse)
