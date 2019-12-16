Original mnist_cnn.py:
^^^^^^^^^^^^^^^^^^^^^^^
The mnist_cnn.py example is located under examples directory of the git
repository https://github.com/fchollet/keras.

In the past Keras was maintained outside of Tensorflow and now it is
integrated in it. Thus if we clone a keras example from a stand-alone
repository and run it we will end up getting run-time errors:

Traceback (most recent call last):
  File "mnist_cnn.py", line 9, in <module>
    import keras
ImportError: No module named keras

We get this error as we haven't installed keras python packages. Since keras
is now integrated in Tensorflow there is no point in installing keras
packages outside Tensorflow.

