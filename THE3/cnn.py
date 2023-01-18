import matplotlib.pyplot as plt
import torch.optim as optim
from utils import *


def train_model(model, index):
    early_stopper = EarlyStopper(patience=2, min_delta=0.0001)

    device = torch.device(DEVICE_ID)  # set device
    model.to(device=device)

    # set loss function ------- if you want to use margin loss, comment the following line and uncomment the line after it
    criterion = nn.MSELoss()
    # criterion = MarginLoss()

    # set optimizers lr parameter to models learning rate
    optimizer = optim.SGD(model.parameters(), lr=model.learning_rate)

    # get data loaders
    train_loader, val_loader = get_loaders(batch_size, device)

    tmp = 0.0
    for epoch in range(max_num_epoch):
        running_loss = 0.0
        plot_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            # loss = criterion(outputs, labels)

            loss = criterion(outputs, labels)
            # loss.backward()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            plot_loss += loss.item()
            print_n = 100
            if i % print_n == print_n - 1:  # print every print_n mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / print_n))
                running_loss = 0.0
            if i == 0 and VISUALIZE:
                # visualize the input, prediction and the output from the last batch
                hw3utils.visualize_batch(inputs, outputs, labels)

        plot_loss /= len(train_loader)
        train_loss.append(plot_loss)
        print('Model %d: Epoch: %d, Train Loss: %.3f' % (index, epoch + 1, plot_loss))

        with torch.no_grad():  # validation

            validation_loss = 0.0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

            tmp = validation_loss / len(val_loader)
            val_loss_array.append(tmp)
            print('Model %d: Epoch: %d, Validation Loss: %f' % (index, epoch + 1, tmp))
            if early_stopper.early_stop(validation_loss):
                print('Early stopping!')
                break

        # print('Saving the model, end of epoch %d' % (epoch + 1))
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        hw3utils.visualize_batch(inputs, outputs, labels,
                                 os.path.join(LOG_DIR, 'model_%d_epoch_%d.png' % (models.index(model), epoch + 1)))

    torch.save(model.state_dict(),
               os.path.join(LOG_DIR, 'model_%d_epoch_%d.pth' % (models.index(model), epoch + 1)))
    print('Finished Training, Validation loss: %.3f of Model %d' % (tmp, models.index(model) + 1))


if __name__ == '__main__':

    DEVICE_ID = 'cuda'  # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
    LOG_DIR = 'checkpoints'
    VISUALIZE = False  # set True to visualize input, prediction and the output from the last batch
    LOAD_CHKPT = False
    torch.multiprocessing.set_start_method('spawn', force=True)
    max_num_epoch = 100
    batch_size = 16
    hyperparameters = [[1, 2, 0.1], [1, 2, 0.0001], [1, 2, 0.001], [4, 2, 0.001], [1, 8, 0.001]]

    train_loss = []
    val_loss_array = []

    models = []
    for number_of_conv_layer, number_of_kernel, learning_rates in hyperparameters:
        models.append(Net(number_of_conv_layer, number_of_kernel, learning_rates))

    for model in models:  # train models
        train_model(model, models.index(model))

    # remove the other models except the first one and
    # uncomment the following lines to plot the train and validation loss for best model
    '''
    # create plot for loss where x-axis is epoch and y-axis is loss of that epoch
    fig, ax = plt.subplots()
    ax.set(xlabel='epoch', ylabel='loss', title='Loss vs Epoch')
    ax.grid()
    plt.xticks(np.arange(0, max_num_epoch, 1))

    
    
    plt.clf()  # clear the plot
    
     # create the plot of the validation 12-margin error using evaluate.py
    fig2, ax2 = plt.subplots()
    ax2.set(xlabel='epoch', ylabel='validation 12-margin error', title='Validation 12-margin error vs Epoch')
    ax2.grid()
    # using loss_array to plot the validation 12-margin error
    for i in range(len(val_loss_array)):
        ax2.plot(i + 1, val_loss_array[i], 'ro')
    fig2.savefig('validation_loss.png')

    plt.clf()  # clear the plot

    # create the plot of the training loss using evaluate.py
    fig3, ax3 = plt.subplots()
    ax3.set(xlabel='epoch', ylabel='training loss', title='Training loss vs Epoch')
    ax3.grid()
    # using loss_array to plot the validation 12-margin error
    for i in range(len(train_loss)):
        ax3.plot(i + 1, train_loss[i], 'ro')
    fig3.savefig('training_loss.png')
    '''

    print('Finished Training: All Models')

    # uncomment the following lines to evaluate the performance of the best model -->
    # for test part to obtain estimations.npy
    # read_model()