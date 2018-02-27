import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer

# def mandel(x, y, max_iters):
#     """
#     Given the real and imaginary parts of a complex number,
#     determine if it is a candidate for membership in the Mandelbrot
#     set given a fixed number of iterations.
#     """
#     c = complex(x, y)
#     z = 0.0j
#     for i in range(max_iters):
#         z = z*z + c
#         if (z.real*z.real + z.imag*z.imag) >= 4:
#             return i
#
#     return max_iters
#
#
# def create_fractal(min_x, max_x, min_y, max_y, image, iters):
#     height = image.shape[0]
#     width = image.shape[1]
#
#     pixel_size_x = (max_x - min_x) / width
#     pixel_size_y = (max_y - min_y) / height
#
#     for x in range(width):
#         real = min_x + x * pixel_size_x
#         for y in range(height):
#             imag = min_y + y * pixel_size_y
#             color = mandel(real, imag, iters)
#             image[y, x] = color
#
#
# image = np.zeros((1024, 1536), dtype = np.uint8)
# start = timer()
# create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
# dt = timer() - start
#
# print("Mandelbrot created in %f s" % dt)
# imshow(image)
# show()


# 第二种 ******************************************************
#
# from numba import autojit
#
#
# @autojit
# def mandel(x, y, max_iters):
#     """
#       Given the real and imaginary parts of a complex number,
#       determine if it is a candidate for membership in the Mandelbrot
#       set given a fixed number of iterations.
#     """
#     c = complex(x, y)
#     z = 0.0j
#     for i in range(max_iters):
#         z = z * z + c
#         if (z.real * z.real + z.imag * z.imag) >= 4:
#             return i
#
#     return max_iters
#
#
# @autojit
# def create_fractal(min_x, max_x, min_y, max_y, image, iters):
#     height = image.shape[0]
#     width = image.shape[1]
#
#     pixel_size_x = (max_x - min_x) / width
#     pixel_size_y = (max_y - min_y) / height
#
#     for x in range(width):
#         real = min_x + x * pixel_size_x
#         for y in range(height):
#             imag = min_y + y * pixel_size_y
#             color = mandel(real, imag, iters)
#             image[y, x] = color
#
#
# image = np.zeros((1024, 1536), dtype=np.uint8)
# start = timer()
# create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
# dt = timer() - start
#
# print("Mandelbrot created in %f s" % dt)
# imshow(image)
# show()


# 第三种 ******************************************************
# from numba import cuda
# from numba import *
#
# def mandel(x, y, max_iters):
#     """
#       Given the real and imaginary parts of a complex number,
#       determine if it is a candidate for membership in the Mandelbrot
#       set given a fixed number of iterations.
#     """
#     c = complex(x, y)
#     z = 0.0j
#     for i in range(max_iters):
#         z = z * z + c
#         if (z.real * z.real + z.imag * z.imag) >= 4:
#             return i
#
#     return max_iters
#
# mandel_gpu = cuda.jit(device=True)(mandel)
#
# @cuda.jit
# def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
#     height = image.shape[0]
#     width = image.shape[1]
#
#     pixel_size_x = (max_x - min_x) / width
#     pixel_size_y = (max_y - min_y) / height
#
#     startX, startY = cuda.grid(2)
#     gridX = cuda.gridDim.x * cuda.blockDim.x
#     gridY = cuda.gridDim.y * cuda.blockDim.y
#
#     for x in range(startX, width, gridX):
#         real = min_x + x * pixel_size_x
#     for y in range(startY, height, gridY):
#         imag = min_y + y * pixel_size_y
#         image[y, x] = mandel_gpu(real, imag, iters)
#
#
# gimage = np.zeros((1024, 1536), dtype=np.uint8)
# blockdim = (40, 8)   # (32, 8)
# griddim = (32, 32)  # (32, 16)
# #
# start = timer()
# d_image = cuda.to_device(gimage)
# mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20)
# d_image.to_host()
# dt = timer() - start
#
# print("Mandelbrot created on GPU in %f s" % dt)
#
# imshow(gimage)
# show()

#
# from numba import cuda
# def prn_obj(obj):
#     print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))
# my_gpu = cuda.get_current_device()
# print(my_gpu.name)    # 获得型号：
# print(my_gpu.COMPUTE_CAPABILITY)   # 获得 Compute Capability
# # print(my_gpu.MUTIPROCESSOR_COUNT)  # 获得SM数量
# # print(my_gpu.MUTIPROCESSOR_COUNT * 128)  # 获得CUDA core的总数
# prn_obj(my_gpu.primary_context)


import pycuda
import pycuda.driver as drv
import numpy as np
from timeit import default_timer as timer

from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void func(float *a, float *b, size_t N)
{
 const int i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i >= N)
 {
  return;
 }
 float temp_a = a[i];
 float temp_b = b[i];
 a[i] = (temp_a * 10 + 2 ) * ((temp_b + 2) * 10 - 5 ) * 5;
 // a[i] = a[i] + b[i];
}
""")

func = mod.get_function("func")


def test(N):
    # N = 1024 * 1024 * 90  # float: 4M = 1024 * 1024

    print("N = %d" % N)

    N = np.int32(N)

    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    # copy a to aa
    aa = np.empty_like(a)
    aa[:] = a
    # GPU run
    nTheads = 256
    nBlocks = int((N + nTheads - 1) / nTheads)
    start = timer()
    func(
        drv.InOut(a), drv.In(b), N,
        block=(nTheads, 1, 1), grid=(nBlocks, 1))
    run_time = timer() - start
    print("gpu run time %f seconds " % run_time)
    # cpu run
    start = timer()
    aa = (aa * 10 + 2) * ((b + 2) * 10 - 5) * 5
    run_time = timer() - start

    print("cpu run time %f seconds " % run_time)

    # check result
    r = a - aa
    print(min(r), max(r))


def main():
    for n in range(1, 10):
        N = 1024 * 1024 * (n * 10)
        print("------------%d---------------" % n)
        test(N)


if __name__ == '__main__':
    main()