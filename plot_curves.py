import utils

import os
import numpy as np
import matplotlib.pyplot as plt

def draw_plot(save_path):
    # load log
    train_log = utils.Logger(os.path.join(save_path,'train.log'))
    test_log = utils.Logger(os.path.join(save_path, 'test.log'))
    # read log
    train_log = train_log.read()
    test_log = test_log.read()
    # zip log data
    epoch, train_loss, train_acc = zip(*train_log)
    epoch, test_loss, test_acc = zip(*test_log)

    # train & valid acc
    plt.plot(epoch, train_acc, '-b', label='train')
    plt.plot(epoch, test_acc, '-r', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.title('TEST accuracy ')
    plt.savefig(os.path.join(save_path, 'test_accuracy.png'))
    plt.close()


    # train entropy loss & ranking loss
    plt.plot(epoch, train_loss, '-b', label='train')
    plt.plot(epoch, test_loss, '-r', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title('loss')
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.close()
