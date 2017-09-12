from keras.datasets import mnist
from keras.utils import np_utils
import numpy


def get_range(xs, rng):
    n = len(xs)
    return [xs[i % n] for i in rng]


def generator_generator(batch_size, X, Y, I, l_indices, u_indices, aug=False):

    # DUMMY = numpy.zeros((batch_size,))
    V = numpy.ones((batch_size, 1)) / batch_size

    for cx in range(200000):

        batch_l_indices = get_range(l_indices, range(cx * batch_size, (cx + 1) * batch_size))
        batch_u_indices = get_range(u_indices, range(cx * batch_size, (cx + 1) * batch_size))
        x_l = numpy.array(get_range(X, batch_l_indices))
        x_u = numpy.array(get_range(X, batch_u_indices))
        y_l = numpy.array(get_range(Y, batch_l_indices))
        # y_u = numpy.array(get_range(y_train, batch_u_indices))
        i_l = get_range(I, batch_l_indices)
        # i_u = get_range(i_train, batch_u_indices)

        if aug:
            div = 0.2
            x_l += numpy.random.randn(*x_l.shape) * div
            x_u += numpy.random.randn(*x_u.shape) * div

        klass_count = [0.001] * 10
        for i in range(10):
            klass = int(i_l[i])
            klass_count[klass] += 1

        T = numpy.array(
            [[1 / klass_count[int(i_l[i])] if int(i_l[i]) == int(i_l[j]) else 0
                for j in range(batch_size)]
                for i in range(batch_size)])

        yield [x_l, x_u], [T, V, y_l]


def batch_generator(labels, unlabels, batch_size=50, aug=False):

    (x_train, i_train), (x_test, i_test) = mnist.load_data()
    x_train = x_train.astype('f') / 255.0
    x_test = x_test.astype('f') / 255.0
    y_train = np_utils.to_categorical(i_train, 10)
    y_test = np_utils.to_categorical(i_test, 10)

    def p(a: str) -> [int]:
        if a is None:
            return [1e20] * 10
        if ',' in a:
            return list(map(int, a.split(',')))
        n = int(a)
        m = n // 10
        return [m] * 10

    # train labeling
    l_indices = []
    u_indices = []
    l_count = [0] * 10
    u_count = [0] * 10
    l_sup = p(labels)
    u_sup = p(unlabels)

    for i in range(len(x_train)):
        klass = int(i_train[i])
        if l_count[klass] < l_sup[klass]:
            l_indices.append(i)
            l_count[klass] += 1
        elif u_count[klass] < u_sup[klass]:
            u_indices.append(i)
            u_count[klass] += 1

    assert len(l_indices) > 0, 'No items are labeeld. see --labels'
    assert len(u_indices) > 0, 'No items are unlabeled. see --unlabels'
    print("{} items are labeled".format(len(l_indices)))
    print("rest {} item are unlabeled".format(len(u_indices)))
    gen_train = generator_generator(batch_size, x_train, y_train, i_train, l_indices, u_indices, aug=aug)

    # dummy labeling for test dataset
    m = len(i_test)
    l_indices = list(range(m))
    u_indices = list(range(batch_size, m)) + list(range(batch_size))
    gen_test = generator_generator(batch_size, x_test, y_test, i_test, l_indices, u_indices)

    return gen_train, gen_test
