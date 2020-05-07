
bool saxpy_cuda(int N, float alpha, float* xarray, float* yarray,
                float* resultarray);
bool matmul_cuda(int rows, int cols, int inner, float* xarray, float* yarray,
                 float* resultarray);