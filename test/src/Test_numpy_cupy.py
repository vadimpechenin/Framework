"""
Просто сравнение производительности cupy и numpy
AMD Ryzen 7 2700 Eight CPU
NVIDIA GeForce GTX 1080 Ti GPU
32 GB of DDR4 3200MHz RAM
Сuda 10.1,
"""

import numpy as np
import cupy as cp
import time

#Уменьшение с 3.495 с до 0.578
### Numpy and CPU
s = time.time()
x_cpu = np.ones((1000,1000,1000))
e = time.time()
print("Numpy ones: " + str(e - s))
### CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000,1000,1000))
cp.cuda.Stream.null.synchronize()
e = time.time()
print("Cupy ones: " + str(e - s))


#Уменьшение с 0.953 с до  0.0625
### Numpy and CPU
s = time.time()
x_cpu *= 5
e = time.time()
print("Numpy *5: " + str(e - s))
### CuPy and GPU
s = time.time()
x_gpu *= 5
cp.cuda.Stream.null.synchronize()
e = time.time()
print("Cupy *5: " + str(e - s))


#Уменьшение с 2.771 с до 0.769
### Numpy and CPU
s = time.time()
x_cpu *= 5
x_cpu *= x_cpu
x_cpu += x_cpu
e = time.time()
print("Numpy +-: " + str(e - s))
### CuPy and GPU
s = time.time()
x_gpu *= 5
x_gpu *= x_gpu
x_gpu += x_gpu
cp.cuda.Stream.null.synchronize()
e = time.time()
print("Cupy +- " + str(e - s))