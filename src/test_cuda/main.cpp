#include <stdio.h>

#include "CycleTimer.h"

int main(void) {
    int N = 1 << 20;
    const float alpha = 2.0f;
    float* xarray = new float[N];
    float* yarray = new float[N];
    float* resultarray = new float[N];

    for (int i = 0; i < N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
    }

    // start time for all operations
    double all_start = CycleTimer::currentSeconds();

    for (int i = 0; i < N; i++) {
        resultarray[i] = alpha * xarray[i] + yarray[i];
    }

    // end time for all operations
    double all_end = CycleTimer::currentSeconds();

    // print time elapse
    double all_duration = all_end - all_start;
    printf("All:\t%.3f ms\n", 1000.f * all_duration);

    delete[] xarray;
    delete[] yarray;
    delete[] resultarray;

    return 0;
}
