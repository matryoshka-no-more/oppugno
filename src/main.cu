#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void 
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
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

