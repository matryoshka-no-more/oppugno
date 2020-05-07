#include <stdio.h>
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

bool saxpy_cuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

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
        return false;
    }

    // free memory buffers on the GPU
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);

    return true;
}

__global__ void 
matmul_kernel(int rows, int cols, int inner, float* x, float* y, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows || j >= cols) return;
    float sum = 0;
    for (int k = 0; k < inner; k++) {
        sum += x[i * inner + k] * y[k * cols + j];
    }
    result[i * cols + j] = sum;
}

bool matmul_cuda(int rows, int cols, int inner, float* xarray, float* yarray, float* resultarray) {

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (max(max(rows, cols), inner) + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;

    int size_x = rows * inner;
    int size_y = cols * inner;
    int size_result = rows * cols;

    // allocate device memory buffers on the GPU using cudaMalloc
    cudaMalloc((void **) &device_x, size_x * sizeof(float));
    cudaMalloc((void **) &device_y, size_y * sizeof(float));
    cudaMalloc((void **) &device_result, size_result * sizeof(float));

    // copy input arrays to the GPU using cudaMemcpy
    cudaMemcpy(device_x, xarray, size_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, size_y * sizeof(float), cudaMemcpyHostToDevice);

    // run kernel
    matmul_kernel<<<blocks, threadsPerBlock>>>(rows, cols, inner, device_x, device_y, device_result);
    cudaDeviceSynchronize();

    // copy result from GPU using cudaMemcpy
    cudaMemcpy(resultarray, device_result, size_result * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        return false;
    }

    // free memory buffers on the GPU
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);

    return true;
}