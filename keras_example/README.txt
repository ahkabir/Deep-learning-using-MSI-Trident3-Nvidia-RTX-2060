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

Mofified mnist_cnn.py (working):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Check the diff from the original mnist_cnn.py that I had to make to
get it to work. Primarily, the changes made were using import from keras
within tensorflow as opposed to assuming we have stand-alone Keras packages.

The log from a successful run is shown below. If you notice the log carefully you
will see that for MNIST training GPU was indeed used. Also you can safely ignore theis warning
"...successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero"

LOG:
====

2019-12-16 16:15:09.970789: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-12-16 16:15:09.972167: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2019-12-16 16:15:09.992541: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:09.992793: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce RTX 2060 major: 7 minor: 5 memoryClockRate(GHz): 1.71
pciBusID: 0000:01:00.0
2019-12-16 16:15:09.992905: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.2
2019-12-16 16:15:09.993778: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2019-12-16 16:15:09.994651: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2019-12-16 16:15:09.994783: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2019-12-16 16:15:09.995734: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2019-12-16 16:15:09.996136: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2019-12-16 16:15:09.997992: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2019-12-16 16:15:09.998092: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:09.998488: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:09.998734: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2019-12-16 16:15:09.998754: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.2
2019-12-16 16:15:10.055748: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-12-16 16:15:10.055771: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2019-12-16 16:15:10.055775: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2019-12-16 16:15:10.055878: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.056122: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.056347: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.056564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5265 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2060, pci bus id: 0000:01:00.0, compute capability: 7.5)
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
2019-12-16 16:15:10.294118: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.294415: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce RTX 2060 major: 7 minor: 5 memoryClockRate(GHz): 1.71
pciBusID: 0000:01:00.0
2019-12-16 16:15:10.294458: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.2
2019-12-16 16:15:10.294482: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2019-12-16 16:15:10.294502: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2019-12-16 16:15:10.294508: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2019-12-16 16:15:10.294530: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2019-12-16 16:15:10.294550: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2019-12-16 16:15:10.294573: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2019-12-16 16:15:10.294644: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.294954: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.295201: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2019-12-16 16:15:10.295412: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.295655: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce RTX 2060 major: 7 minor: 5 memoryClockRate(GHz): 1.71
pciBusID: 0000:01:00.0
2019-12-16 16:15:10.295667: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.2
2019-12-16 16:15:10.295689: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2019-12-16 16:15:10.295696: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2019-12-16 16:15:10.295703: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2019-12-16 16:15:10.295723: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2019-12-16 16:15:10.295729: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2019-12-16 16:15:10.295750: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2019-12-16 16:15:10.295788: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.296020: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.296216: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2019-12-16 16:15:10.296248: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-12-16 16:15:10.296252: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2019-12-16 16:15:10.296256: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2019-12-16 16:15:10.296313: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.296542: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-12-16 16:15:10.296801: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5265 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2060, pci bus id: 0000:01:00.0, compute capability: 7.5)
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
2019-12-16 16:15:11.163911: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2019-12-16 16:15:11.323032: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
60000/60000 [==============================] - 5s 89us/sample - loss: 0.6819 - accuracy: 0.7772 - val_loss: 0.2067 - val_accuracy: 0.9360
Epoch 2/12
60000/60000 [==============================] - 4s 66us/sample - loss: 0.4394 - accuracy: 0.8613 - val_loss: 0.1645 - val_accuracy: 0.9496
Epoch 3/12
60000/60000 [==============================] - 3s 57us/sample - loss: 0.3262 - accuracy: 0.8995 - val_loss: 0.1089 - val_accuracy: 0.9656
Epoch 4/12
60000/60000 [==============================] - 4s 59us/sample - loss: 0.2724 - accuracy: 0.9170 - val_loss: 0.0926 - val_accuracy: 0.9710
Epoch 5/12
60000/60000 [==============================] - 4s 63us/sample - loss: 0.2683 - accuracy: 0.9185 - val_loss: 0.0967 - val_accuracy: 0.9712
Epoch 6/12
60000/60000 [==============================] - 4s 69us/sample - loss: 0.2462 - accuracy: 0.9268 - val_loss: 0.1010 - val_accuracy: 0.9673
Epoch 7/12
60000/60000 [==============================] - 4s 61us/sample - loss: 0.2398 - accuracy: 0.9267 - val_loss: 0.0907 - val_accuracy: 0.9739
Epoch 8/12
60000/60000 [==============================] - 4s 68us/sample - loss: 0.2263 - accuracy: 0.9322 - val_loss: 0.0896 - val_accuracy: 0.9726
Epoch 9/12
60000/60000 [==============================] - 3s 56us/sample - loss: 0.2238 - accuracy: 0.9326 - val_loss: 0.0912 - val_accuracy: 0.9728
Epoch 10/12
60000/60000 [==============================] - 4s 61us/sample - loss: 0.2172 - accuracy: 0.9341 - val_loss: 0.0815 - val_accuracy: 0.9762
Epoch 11/12
60000/60000 [==============================] - 4s 68us/sample - loss: 0.2189 - accuracy: 0.9344 - val_loss: 0.0787 - val_accuracy: 0.9760
Epoch 12/12
60000/60000 [==============================] - 4s 65us/sample - loss: 0.2098 - accuracy: 0.9367 - val_loss: 0.0844 - val_accuracy: 0.9733
Test loss: 0.08442770325504244
Test accuracy: 0.9733
