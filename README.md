# Learning by association; Semi-supervised MNIST learning

This is an implementation of "Learning by Assocication" written with Keras.
`assoc.py` learns MNIST, 100 (10x10) labeled and unlabeled data.

## MNIST

論文や実装 (https://github.com/haeusser/learning_by_association/blob/master/semisup/mnist_train_eval.py) では数分で99%を達成するとあるが、全然達成できなかった.
`assoc.py` では 22 epochs 回してようやく 96.6% 程度に到達しました.

## References

- The original paper: [[1706.00909] Learning by Association - A versatile semi-supervised training method for neural networks](https://arxiv.org/abs/1706.00909)
- Source code by them: [https://github.com/haeusser/learning_by_association](https://github.com/haeusser/learning_by_association)
- 日本語の解説: [https://github.com/arXivTimes/arXivTimes/issues/352](https://github.com/arXivTimes/arXivTimes/issues/352)
