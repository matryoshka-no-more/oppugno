#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import ast
import shlex
import inspect
# import astpretty
import subprocess
from random import random
from time import time
from prettytable import PrettyTable

PATH_CURR = os.path.dirname(os.path.realpath(__file__))
PATH_BASE = os.path.dirname(PATH_CURR)
sys.path.append(PATH_BASE)

from oppugno.cuda import Cuda

TABLE_HEADERS = ["Array Size", "GPU speedup"]
DEFAULT_ARR_SIZE = 1 << 10
ALPHA = 2.0
execname = os.path.join(os.getcwd(), "main")
filename = os.path.join(os.getcwd(), "main.cu")
str_cuda_saxpy = """#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void 
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= N) return;
    for (int i = 0; i < N; i++) {
        result[index] = alpha * x[index] + y[index];
    }
}

int main(void) {
    int N;
    float alpha;
    cin >> N;
    cin >> alpha;
    
    float* xarray = new float[N];
    float* yarray = new float[N];
    float* resultarray = new float[N];

    for (int i=0; i<N; i++) {
        cin >> xarray[i];
    }
    for (int i=0; i<N; i++) {
        cin >> yarray[i];
    } 
    for (int i=0; i<N; i++) {
        resultarray[i] = 0.f;
    }

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;

    // allocate device memory buffers on the GPU using cudaMalloc
    cudaMalloc((void **) &device_x, N * sizeof(float));
    cudaMalloc((void **) &device_y, N * sizeof(float));
    cudaMalloc((void **) &device_result, N * sizeof(float));

    // copy input arrays to the GPU using cudaMemcpy
    cudaMemcpy(device_x, xarray, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, N * sizeof(float), cudaMemcpyHostToDevice);

    // run kernel
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaDeviceSynchronize();

    // copy result from GPU using cudaMemcpy
    cudaMemcpy(resultarray, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        return 1;
    }

    for (int i = 0; i < N; i++) {
        cout << resultarray[i] << " ";
    }
    cout << "" << endl;

    // free memory buffers on the GPU
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);

    delete [] xarray;
    delete [] yarray;
    delete [] resultarray;

    return 0;
}
"""


def time_func(method):
    def func(*args, **kw):
        start = time()
        result = method(*args, **kw)
        stop = time()
        duration = stop - start
        print("{}: {:.3f}ms".format(method.__name__, duration * 1000))
        return duration, result

    return func


def parse_func(method):
    def func(*args, **kw):
        code_str = inspect.getsource(method)
        code_ast = ast.parse(code_str)
        print(code_str)
        # astpretty.pprint(code_ast)
        result = method(*args, **kw)
        return result

    return func


def get_data(size=DEFAULT_ARR_SIZE):
    data = [0] * size
    for i in range(size):
        data[i] = random()
    return data


@time_func
def saxpy_cpu(data_x, data_y, alpha=ALPHA):
    size = len(data_x)
    result = [0.0] * size
    for i in range(size):
        for _ in range(size):
            result[i] = alpha * data_x[i] + data_y[i]
    return result


def compile_cuda(func):
    with open(filename, "w+") as f:
        f.write(func)
    cmd = shlex.split("nvcc -o {} {}".format(execname, filename))
    rc = subprocess.call(cmd)
    return rc == 0


@time_func
def saxpy_gpu(data_x, data_y, alpha=ALPHA):
    # compile cuda
    if not compile_cuda(str_cuda_saxpy):
        print("Error: unable to compile CUDA file")
        return

    size = len(data_x)
    cmd = shlex.split(execname)
    args = "{}\n{}\n{}\n{}".format(size, ALPHA,
                                   " ".join([str(x) for x in data_x]),
                                   " ".join([str(y) for y in data_y]))
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = p.communicate(str.encode(args))
    stdout_lines = stdout.decode("utf-8").strip().split("\n")
    result = [float(x) for x in stdout_lines[0].split(" ")]
    return result


def compare(N=DEFAULT_ARR_SIZE):
    print("========== compare array size: {} ==========".format(N))
    data_x = get_data(N)
    data_y = get_data(N)
    time_cpu, result_cpu = saxpy_cpu(data_x, data_y)
    time_gpu, result_gpu = saxpy_gpu(data_x, data_y)

    for i in range(len(result_cpu)):
        if result_cpu[i] - result_gpu[i] > 0.00001:
            print("Failed: [{}]\t(cpu){:4f} != {:4f}(gpu)".format(
                i, result_cpu[i], result_gpu[i]))
            return time_cpu, time_gpu
    print("Success: the results from cpu and gpu are consistent")
    return time_cpu, time_gpu


def main():
    t = PrettyTable(TABLE_HEADERS)
    t.align[TABLE_HEADERS[0]] = "l"
    t.align[TABLE_HEADERS[1]] = "r"
    for i in range(2, 16):
        N = 1 << i
        time_cpu, time_gpu = compare(N)
        t.add_row([N, "{:4f}".format(time_cpu / time_gpu)])

    print("\n========== Summary ==========\n")
    print(t)


if __name__ == "__main__":
    main()