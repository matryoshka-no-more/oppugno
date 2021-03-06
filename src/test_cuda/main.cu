#include <stdio.h>
#include "CycleTimer.h"

__global__ void 
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

int main(void) {
    int N = 1<<20;
    const float alpha = 2.0f;
    float* xarray = new float[N];
    float* yarray = new float[N];
    float* resultarray = new float[N];
    
    for (int i=0; i<N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
    }

    // start time for all operations
    double all_start = CycleTimer::currentSeconds();

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

    // start timing after allocation of device memory
    double cuda_start = CycleTimer::currentSeconds();

    // run kernel
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaDeviceSynchronize();

    // end timing after result has been copied back into host memory
    double cuda_end = CycleTimer::currentSeconds();

    // copy result from GPU using cudaMemcpy
    cudaMemcpy(resultarray, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // free memory buffers on the GPU
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);

    // end time for all operations
    double all_end = CycleTimer::currentSeconds();

    // print time elapse
    double cuda_duration = cuda_end - cuda_start;
    double all_duration = all_end - all_start;
    printf("All:\t%.3f ms\nCUDA:\t%.3f ms\n", 1000.f * all_duration, 1000.f * cuda_duration);

    delete [] xarray;
    delete [] yarray;
    delete [] resultarray;

    return 0;
}

