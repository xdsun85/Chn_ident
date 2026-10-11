from keras import models, layers    # , utils

n_ts = 32
n_feat = 1
n_cls = 3

def vNN(input_shape=(n_ts,), output_shape=n_cls, sizes=[64, 64]):
    n_hidden_layers = len(sizes)
    myNet = models.Sequential()
    myNet.add(layers.Dense(units=sizes[0], activation='relu', name='F1', input_shape=input_shape))
    for i_hl in range(n_hidden_layers - 1):
        myNet.add(layers.Dense(units=sizes[i_hl + 1], activation='relu', name='F%s' % (i_hl + 2)))
    myNet.add(layers.Dense(units=output_shape, activation='softmax', name='F%s' % (n_hidden_layers + 1)))

    # myNet.summary()

    return myNet

def odCNN(input_shape=(n_ts, n_feat), output_shape=n_cls, n_filters=64, k_size=16):
    myNet = models.Sequential()
    myNet.add(layers.Conv1D(filters=n_filters, kernel_size=k_size, strides=1, activation='relu', name='C1', input_shape=input_shape))
    # print('C1 output_shape', myNet.output_shape)
    myNet.add(layers.Conv1D(filters=n_filters, kernel_size=k_size, strides=1, activation='relu', name='C2'))
    myNet.add(layers.Dropout(0.5))
    myNet.add(layers.MaxPooling1D(pool_size=2, strides=2, name='P3'))
    myNet.add(layers.Flatten(name='F4'))
    myNet.add(layers.Dense(units=100, activation='relu', name='F5'))
    myNet.add(layers.Dense(units=output_shape, activation='softmax', name='F6'))

    # myNet.summary()

    return myNet
