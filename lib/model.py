from keras.layers import Dense, Flatten, Reshape, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K

from lib.losses import walk_loss, visit_loss, classification_loss


SHAPE = (28, 28)


def Embedding():
    model = Sequential(name='embedding')
    model.add(Reshape(SHAPE + (1,), input_shape=SHAPE))
    model.add(Conv2D(32, (3, 3), activation='elu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='elu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='elu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='elu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    return model


def calc_walk(args):
    z1, z2 = args
    match = K.dot(z1, K.transpose(z2))  # ((m, Z_DIM), (n, Z_DIM)^t) -> (m, n)
    p = K.softmax(match)  # P^{ab}: (m, n)
    q = K.softmax(K.transpose(match))  # P^{ba}: (n, m)
    walk = K.dot(p, q)
    return walk


def calc_visit(args):
    z1, z2 = args
    match = K.dot(z1, K.transpose(z2))
    p = K.softmax(match)
    visit = K.mean(K.transpose(p), axis=1, keepdims=True)
    return visit


def build():
    xl = Input(shape=SHAPE)
    xu = Input(shape=SHAPE)
    embedding = Embedding()
    predict = Dense(10, name='predict', activation='softmax')
    zl = embedding(xl)
    zu = embedding(xu)
    yl = predict(zl)
    # yu = predict(zu)

    walk = Lambda(calc_walk, name='walk')([zl, zu])
    visit = Lambda(calc_visit, name='visit')([zl, zu])

    opt = Adam(clipvalue=0.01, lr=1e-3, decay=.003)

    model = Model(inputs=[xl, xu], outputs=[walk, visit, yl])
    model.compile(loss=[walk_loss, visit_loss, classification_loss],
                  loss_weights=[1, .5, 1],
                  metrics={'predict': ['acc']},
                  optimizer=opt)

    return model
