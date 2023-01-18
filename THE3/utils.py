from skimage import io, color
from skimage.transform import rescale
import numpy as np
import torch
import torch.nn as nn
import hw3utils
import os


# ---- ConvNet -----
class Net(nn.Module):
    def __init__(self, number_of_conv_layers, number_of_kernels, learning_rate):
        super(Net, self).__init__()
        self.main = None
        self.learning_rate = learning_rate
        if number_of_conv_layers == 1:
            self.main = nn.Sequential(
                conv_block(1, number_of_kernels),
                last_conv_block(number_of_kernels, 3)
            )

        elif number_of_conv_layers == 4:
            self.main = nn.Sequential(
                conv_block(1, number_of_kernels),
                conv_block(number_of_kernels, number_of_kernels),
                last_conv_block(number_of_kernels, 3)
            )

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        out = self.main(grayscale_image)
        return out


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.prev = np.inf

    def early_stop(self, validation_loss):
        validation_loss *= 1000
        if validation_loss < self.prev:
            self.counter = 0
        elif validation_loss > (self.prev + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.prev = validation_loss
                return True
        self.prev = validation_loss
        return False


class MarginLoss(nn.Module):
    def __init__(self, margin=0.094):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, x, y):
        absolute_difference = torch.abs(x - y)
        return torch.mean(torch.max(absolute_difference - self.margin, torch.zeros_like(absolute_difference)))


def conv_block(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, (3, 3), padding=1, groups=1),
        nn.ReLU(),
        nn.MaxPool2d((1, 1))
    )


def last_conv_block(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, (3, 3), padding=1),
        nn.Tanh()
    )


def read_image(filename):
    img = io.imread(filename)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], 2)
    return img


def cvt2Lab(image):
    Lab = color.rgb2lab(image)
    return Lab[:, :, 0], Lab[:, :, 1:]  # L, ab


def cvt2rgb(image):
    return color.lab2rgb(image)


def upsample(image):
    return rescale(image, 4, mode='constant', order=3)


def read_model():
    model = Net(1, 2, 0.1)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # create 100 random numbers between 0 and 1999
    random_numbers = np.random.randint(0, 1999, 100)
    # sort the random numbers
    random_numbers.sort()
    # create a txt file and write 'images_grayscale/#.jpg' to it for each random number
    with open('test.txt', 'w') as f:
        for i in random_numbers:
            f.write('images_grayscale/images/%d.jpg\n' % i)

    # load images from images_grayscale folder using test.txt
    test_set = hw3utils.HW3ImageFolder(root='images_grayscale', device='cpu')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # create empty np array to store the results
    predictions = np.empty((100, 80, 80, 3))
    # for each image in test_loader and random number in random_numbers, save the image and its prediction
    count = 0
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    for i, (img, _) in enumerate(test_loader):
        if i in random_numbers:
            prediction = model(img)

            hw3utils.visualize_batch_for_test(img, prediction, save_path='outputs/output%d.png' % i)
            prediction = torch.permute(prediction, (0, 2, 3, 1))
            # scale the prediction to 0-255
            scaled = (prediction + 1) * 127.5
            predictions[count] = scaled.detach().numpy()
            count += 1
            print('output%d.png saved' % i)

    # save the predictions to a npy file

    np.save('estimations.npy', predictions)
    return


def get_loaders(batch_size,device):
    data_root = 'ceng483-f22-hw3-dataset'
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

