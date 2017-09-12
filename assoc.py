import click
# import keras.callbacks
import tensorflow as tf
from keras import backend as K
import numpy

import dataset
import lib.log
import lib.model


def echo(*args):
    click.secho(' '.join(str(arg) for arg in args), fg='green', err=True)


@click.group()
def main():
    pass


@main.command()
@click.option('--name', help='model name')
@click.option('--labels', default="100")
@click.option('--unlabels', default=None)
@click.option('--aug', is_flag=True, default=False, help='data augmentation every epoch')
@click.option('--resume', help='when resume learning from the snapshot')
def train(name, labels, unlabels, aug, resume):

    numpy.set_printoptions(
        precision=2,
        linewidth=140)

    # paths
    log_path = "logs/{}.json".format(name)
    out_path = "snapshots/" + name + ".{epoch:06d}.h5"
    echo('log path', log_path)
    echo('out path', out_path)

    lib.log.info(log_path, {'_commandline': {
        'name': name, 'labels': labels, 'unlabels': unlabels, 'aug': aug, 'resume': resume}})

    # init
    echo('train', (name, resume))
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(1)

    # dataset
    echo('dataset loading...')
    batch_size = 100
    gen_train, gen_test = dataset.batch_generator(labels, unlabels, batch_size=batch_size, aug=aug)

    # model building
    echo('model building...')
    model = lib.model.build()
    model.summary()
    if resume:
        echo('Resume Learning from {}'.format(resume))
        model.load_weights(resume, by_name=True)

    # x_tensor = model.layers[2].get_input_at(0)
    # z_tensor = model.layers[2].get_output_at(0)
    # embedding = K.function([x_tensor], [z_tensor])
    # z_tensor = model.layers[5].get_input_at(0)
    # y_tensor = model.layers[5].get_output_at(0)
    # predict = K.function([z_tensor], [y_tensor])

    # training
    echo('start learning...')
    callbacks = [
        lib.log.JsonLog(log_path),
        # keras.callbacks.ModelCheckpoint(out_path, monitor='val_loss', save_weights_only=True)
    ]
    model.fit_generator(
        gen_train,
        epochs=50,
        steps_per_epoch=(59900 // batch_size),
        validation_data=gen_test,
        validation_steps=(1000 // batch_size),
        callbacks=callbacks)


if __name__ == '__main__':
    main()
