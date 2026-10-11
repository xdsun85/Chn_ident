import os
import numpy as np
import random
import matplotlib.pyplot as plt

from keras import callbacks

from nn import vNN, odCNN

def normalization(arr):
    arr = np.array(arr)
    if arr.size == 0:          # leer, keep it was
        return arr
    elif (arr == arr[0]).all():   # if there were only one item or all items were the same, output all 0.5 s
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
    # exit()
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

    repeats = 10
    full_acc_list = []

    sizes_params = [[128], [64, 64], [42, 42, 42], [32, 32, 32, 32]]
    for sizes in sizes_params:
        single_acc_list = []
        for rpt in range(repeats):
            print('\nvNN: %s, #%s' % (sizes, rpt))
            # build model: vanilla NN
            myNet = vNN(sizes=sizes)

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
                patience=10,
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
                                  callbacks=[reduce_lr, early_stopping])  # checkpoint_period1

            test_loss, test_acc = myNet.evaluate(x=X_test, y=y_test)
            print('test_loss:', test_loss)
            print('test_acc:', test_acc)
            single_acc_list.append(test_acc)
        full_acc_list.append(['vNN', sizes, single_acc_list])
        print('single_acc_list: %s\n' % single_acc_list)

    n_filters_params = [8, 16, 32, 64, 128, 256]  #
    k_size_params = [3, 5, 7, 11, 13, 15, 16]  # Never greater than 32 / 2 = 16   #
    for n_filters in n_filters_params:
        for k_size in k_size_params:
            single_acc_list = []
            for rpt in range(repeats):
                print('\n1dCNN: n_filters: %s, k_size: %s, #%s' % (n_filters, k_size, rpt))
                # build model: one dimensional CNN
                myNet = odCNN(n_filters=n_filters, k_size=k_size)

                #
                # checkpoint_period1 = callbacks.ModelCheckpoint(
                #     log_dir + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                #     monitor='accuracy',
                #     save_weights_only=False,
                #     save_best_only=True,
                #     save_freq='epoch'  # 'epoch' or integer. interger = N × (train_set_datasize / batch_size / 100)
                # )
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
                    patience=10,
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
                                      callbacks=[reduce_lr, early_stopping])     # checkpoint_period1

                test_loss, test_acc = myNet.evaluate(x=X_test, y=y_test)
                print('test_loss:', test_loss)
                print('test_acc:', test_acc)
                single_acc_list.append(test_acc)
            full_acc_list.append(['1dCNN', [n_filters, k_size], single_acc_list])
            print('single_acc_list: %s\n' % single_acc_list)
    print('full_acc_list = %s\n' % full_acc_list)

    fig = plt.figure(figsize=(21.25, 4), dpi=120)
    plt.boxplot([np.array(i[2]) for i in full_acc_list],
                labels=['%s@%s' % (i[1][0], i[1][1]) if i[0] == '1dCNN' else '%s*%s' % (i[1][0], len(i[1])) for i in full_acc_list],
                showfliers=False, widths=0.6)
    plt.xticks(rotation=50, fontsize=14)
    ax = plt.gca()
    for i, tick_label in enumerate(ax.get_xticklabels()):
        if i <= 3:
            tick_label.set_color('#999900')
        elif 4 <= i <= 10 or 18 <= i <= 24 or 32 <= i <= 38:
            tick_label.set_color('#0077FF')
        elif 11 <= i <= 17 or 25 <= i <= 31 or 39 <= i <= 45:
            tick_label.set_color('C9')

    plt.text(2.5, 0.92, 'vNNs', color='#999900', fontsize=18, horizontalalignment='center')
    plt.text(10, 0.92, '1D-CNNs', color='C0', fontsize=18)

    plt.text(8, 0.69, '8 filters', color='#0077FF', fontsize=14, horizontalalignment='center')
    plt.text(15, 0.69, '16 filters', color='C9', fontsize=14, horizontalalignment='center')
    plt.text(22, 0.69, '32 filters', color='#0077FF', fontsize=14, horizontalalignment='center')
    plt.text(29, 0.69, '64 filters', color='C9', fontsize=14, horizontalalignment='center')
    plt.text(36, 0.69, '128 filters', color='#0077FF', fontsize=14, horizontalalignment='center')
    plt.text(43, 0.69, '256 filters', color='C9', fontsize=14, horizontalalignment='center')

    # x_bd_left, x_bd_right = plt.xlim()
    y_bd_low, y_bd_high = plt.ylim()
    h_rect = y_bd_high - y_bd_low
    myRect_vnn = plt.Rectangle((0, y_bd_low), 4.5, h_rect, color='#FFFF00', fill=True, lw=0.1, alpha=0.1)  # Coordinates of the apex close to the origin, width, height
    ax.add_patch(myRect_vnn)
    myRect_1dcnn_8f = plt.Rectangle((4.5, y_bd_low), 7, h_rect, color='#0077FF', fill=True, lw=0.1, alpha=0.1)  # #1F77FF
    ax.add_patch(myRect_1dcnn_8f)
    myRect_1dcnn_16f = plt.Rectangle((11.5, y_bd_low), 7, h_rect, color='#00FFFF', fill=True, lw=0.1, alpha=0.1)
    ax.add_patch(myRect_1dcnn_16f)
    myRect_1dcnn_32f = plt.Rectangle((18.5, y_bd_low), 7, h_rect, color='#0077FF', fill=True, lw=0.1, alpha=0.1)
    ax.add_patch(myRect_1dcnn_32f)
    myRect_1dcnn_64f = plt.Rectangle((25.5, y_bd_low), 7, h_rect, color='#00FFFF', fill=True, lw=0.1, alpha=0.1)
    ax.add_patch(myRect_1dcnn_64f)
    myRect_1dcnn_128f = plt.Rectangle((32.5, y_bd_low), 7, h_rect, color='#0077FF', fill=True, lw=0.1, alpha=0.1)
    ax.add_patch(myRect_1dcnn_128f)
    myRect_1dcnn_256f = plt.Rectangle((39.5, y_bd_low), 7, h_rect, color='#00FFFF', fill=True, lw=0.1, alpha=0.1)
    ax.add_patch(myRect_1dcnn_256f)

    ax.set_ylim(y_bd_low, y_bd_high)

    plt.xlabel('Network model configurations', fontsize=14, labelpad=-4)
    plt.ylabel('Accuracy on the test set', fontsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.title('Performance comparison among a range of vNN and 1D-CNN models for channel identification', fontsize=16)

    # plt.show()
    plt.savefig('./chn_ident.png', bbox_inches='tight', pad_inches=0.01)
    plt.close()
