#! /usr/bin/python3.6
import numpy as np
import os

os.system('clear')
print('numpy is very crucial to tensorflows')

# np arange
print('np.arange(0, 10) is a range of 0-10')
print(np.arange(0,10))

# np newaxis
print('\n\nwhat is np.newaxis?\nit is used to increase the dimension of existing array by one more dimension when used once')
print(np.newaxis)
print('lets make an np arange of 4')
arr = np.arange(4)
print(arr)
print('look at its shape with "arange.shape":')
print(arr.shape)
print('make it a row vector by inserting an axiss along 1st dimension')
row_vector = arr[np.newaxis, :]
print(row_vector.shape)
print(row_vector)
print('what if we added a column?')
col_vector = arr[:, np.newaxis]
print(col_vector.shape)
print(col_vector)
