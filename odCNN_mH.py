from keras import models, layers    #, utils

n_ts = 32
n_feat = 1
n_cls = 3

def odCNN_mH(input_shape=(n_ts, n_feat), output_shape=n_cls):
    # head 1
    inputs_1 = layers.Input(shape=input_shape, name='I1')
    conv_1 = layers.Conv1D(filters=64, kernel_size=16, strides=1, activation='relu', name='C1')(inputs_1)
    drop_1 = layers.Dropout(0.5)(conv_1)
    pool_1 = layers.MaxPooling1D(pool_size=2, strides=2, name='P1')(drop_1)
    flat_1 = layers.Flatten(name='F1')(pool_1)

    # head 2
    inputs_2 = layers.Input(shape=input_shape, name='I2')
    conv_2 = layers.Conv1D(filters=128, kernel_size=13, strides=1, activation='relu', name='C2')(inputs_2)
    drop_2 = layers.Dropout(0.5)(conv_2)
    pool_2 = layers.MaxPooling1D(pool_size=2, strides=2, name='P2')(drop_2)
    flat_2 = layers.Flatten(name='F2')(pool_2)

    # head 3
    inputs_3 = layers.Input(shape=input_shape, name='I3')
    conv_3 = layers.Conv1D(filters=256, kernel_size=11, strides=1, activation='relu', name='C3')(inputs_3)
    drop_3 = layers.Dropout(0.5)(conv_3)
    pool_3 = layers.MaxPooling1D(pool_size=2, strides=2, name='P3')(drop_3)
    flat_3 = layers.Flatten(name='F3')(pool_3)

    # merge
    merged = layers.merging.concatenate([flat_1, flat_2, flat_3], name='M')
    # interpretation
    dense_1 = layers.Dense(units=100, activation='relu', name='F4')(merged)
    dense_2 = layers.Dense(units=output_shape, activation='softmax', name='F5')(dense_1)

    myNet = models.Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=dense_2)

    myNet.summary()

    # save a plot of the model
    # utils.vis_utils.plot_model(myNet, show_shapes=True, to_file='./multichannel.png')

    return myNet
