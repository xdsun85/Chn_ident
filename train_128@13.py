import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from keras import callbacks

from nn import odCNN

def normalization(arr):
    arr = np.array(arr)
    if arr.size == 0:          # # if leer, keep it was
        return arr
    elif (arr == arr[0]).all():   # if there were only one item or all items were the same, output all 0.5s
        return np.array([0.5] * arr.size)
    else:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def load_data():
    '''load the data for training and validation of all classes'''
    vld = []
    io_vld = open('./sample/valid.txt')
    for line in io_vld:
        spl_str = line.split('#')[0].strip('\n').split(', ')
        if spl_str[0] != '':
            spl_float = np.asfarray(spl_str)
            spl_norm = normalization(spl_float)
            vld.append([np.array([spl_norm]).transpose(), [0, 0, 1]])  # one-hot
    io_vld.close()

    ivld = []
    io_ivld = open('./sample/invalid.txt')
    for line in io_ivld:
        spl_str = line.split('#')[0].strip('\n').split(', ')
        if spl_str[0] != '':
            spl_float = np.asfarray(spl_str)
            spl_norm = normalization(spl_float)
            ivld.append([np.array([spl_norm]).transpose(), [0, 1, 0]])  # one-hot
    io_ivld.close()

    inv = []
    io_inv = open('./sample/inverse.txt')
    for line in io_inv:
        spl_str = line.split('#')[0].strip('\n').split(', ')
        if spl_str[0] != '':
            spl_float = np.asfarray(spl_str)
            spl_norm = normalization(spl_float)
            inv.append([np.array([spl_norm]).transpose(), [1, 0, 0]])  # one-hot
    io_inv.close()
    n_vld, n_ivld, n_inv = len(vld), len(ivld), len(inv)
    print(n_vld, n_ivld, n_inv)  #
    # exit()

    random.shuffle(vld)
    random.shuffle(ivld)
    random.shuffle(inv)

    training_set = vld[: n_ivld - 40] + ivld[: n_ivld - 40] + inv[: n_inv - 20]
    # training_set = vld[: n_vld - 40] + ivld[: n_ivld - 40] + inv[: n_inv - 20]
    validation_set = vld[n_ivld - 40: n_ivld] + ivld[n_ivld - 40: n_ivld] + inv[n_inv - 20: n_inv]
    test_set = vld[n_ivld - 40: n_ivld] + ivld[n_ivld - 40: n_ivld] + inv[n_inv - 20: n_inv]
    # print(len(vld[: n_ivld - 40]), len(ivld[: n_ivld - 40]), len(inv[: n_inv - 20]))
    # print(len(vld[n_ivld - 40: n_ivld]), len(ivld[n_ivld - 40: n_ivld]), len(inv[n_inv - 20: n_inv]))
    # exit()

    X_train, X_test = np.empty((0, 32, 1)), np.empty((0, 32, 1))
    y_train, y_test = np.empty((0, 3)), np.empty((0, 3))
    for spl in training_set:
        X_train = np.append(X_train, [spl[0]], axis=0)
        y_train = np.append(y_train, [spl[1]], axis=0)
    for spl in test_set:
        X_test = np.append(X_test, [spl[0]], axis=0)
        y_test = np.append(y_test, [spl[1]], axis=0)
    print(X_train.shape)        # (, 32, 1)
    print(y_train.shape)        # (, 3)
    print(X_test.shape)         # (100, 32, 1)
    print(y_test.shape)         # (100, 3)

    # return training_set, validation_set, test_set
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    log_dir = './logs'         # the directory for logs and trained model
    if 0 == os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # training_set, validation_set, test_set = load_data()
    X_train, X_test, y_train, y_test = load_data()
    print(X_train.shape)        # (, 32, 1)
    print(X_test.shape)         # (100, 32, 1)
    print(y_train.shape)       # (, 3)
    print(y_test.shape)        # (100, 3)

    n_spl_X_train, n_ts_train, _ = X_train.shape
    n_spl_X_test, n_ts_test, _ = X_test.shape
    n_spl_y_train, n_cls_train = y_train.shape
    n_spl_y_test, n_cls_test = y_test.shape
    if n_spl_X_train != n_spl_y_train:
        err_txt = 'The sample sizes of data and labels in training set are NOT consistent!'
        print('\033[0;31;48m%s\033[0m' % err_txt)
        exit()
    if n_spl_X_test != n_spl_y_test:
        err_txt = 'The sample sizes of data and labels in test set are NOT consistent!'
        print('\033[0;31;48m%s\033[0m' % err_txt)
        exit()
    if n_ts_train != n_ts_test:
        err_txt = 'The timestamps in training and test sets are NOT consistent!'
        print('\033[0;31;48m%s\033[0m' % err_txt)
        exit()
    if n_cls_train != n_cls_test:
        err_txt = 'The class amounts in training and test sets are NOT consistent!'
        print('\033[0;31;48m%s\033[0m' % err_txt)
        exit()

    from sklearn.utils import shuffle
    # To prevent similar samples from clustering and causing training results to depend on the sample order, we have to shuffle the sample order before training.
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # build model: one dimensional CNN
    myNet = odCNN(n_filters=128, k_size=13)

    #
    checkpoint_period1 = callbacks.ModelCheckpoint(
        log_dir + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='accuracy',
        save_weights_only=False,
        save_best_only=True,
        save_freq='epoch'  # 'epoch' or integer. interger = N × (train_set_datasize / batch_size / 100)
    )
    #
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        verbose=1
    )
    #
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1
    )

    myNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    n_epc = 100
    batch_size = 128
    # fit network
    myHistory = myNet.fit(x=X_train, y=y_train,
                          epochs=n_epc,
                          batch_size=batch_size,
                          validation_split=0.1,
                          verbose=1,
                          callbacks=[reduce_lr, early_stopping])    # checkpoint_period1,

    from matplotlib.pyplot import MultipleLocator

    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(0.2)

    plt.figure(figsize=(6.3, 5), dpi=100)
    plt.subplots_adjust(hspace=0.65)
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.plot(myHistory.history['accuracy'], label='Acc', lw=1)
    plt.plot(myHistory.history['loss'], label='Loss', lw=1)
    plt.title('Acc, loss on the training set', fontsize=18)
    plt.xlabel('Epoch', fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.ylabel('Ratio', fontsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.axis([-2, 57, 0.10, 1])
    plt.legend(loc='center right', fontsize=16)
    print(len(myHistory.history['accuracy']))
    print('acc = %s' % myHistory.history['accuracy'])
    print('loss = %s' % myHistory.history['loss'])

    plt.subplot(2, 1, 2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.plot(myHistory.history['val_accuracy'], label='Acc', lw=1)
    plt.plot(myHistory.history['val_loss'], label='Loss', lw=1)
    plt.title('Acc, loss on the validation set', fontsize=18)
    plt.xlabel('Epoch', fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.ylabel('Ratio', fontsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.axis([-2, 57, 0.10, 1])
    plt.legend(loc='center right', fontsize=16)
    print('val_acc = %s' % myHistory.history['val_accuracy'])
    print('val_loss = %s' % myHistory.history['val_loss'])

    plt.savefig(log_dir + '/Eval_128@13.png', bbox_inches='tight', pad_inches=0.02)
    plt.close()

    test_loss, test_acc = myNet.evaluate(x=X_test, y=y_test)
    print('test_loss:', test_loss)
    print('test_acc:', test_acc)

    myNet.save_weights(log_dir + '/odCNN_128@13_weights.h5')
