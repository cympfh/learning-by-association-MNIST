classification_loss = 'categorical_crossentropy'
walk_loss = 'binary_crossentropy'
visit_loss = 'binary_crossentropy'


def zero_loss(y_true, y_pred):
    return y_true * 0.0
