// src/lstm/cpp/slstm.cpp
#include <iostream>
#include <vector>
#ifdef USE_CUDA
#include <cuda_runtime.h>
extern void slstm_forward_cuda(float* input, float* hidden_state, float* cell_state, float* weights, float* output, int batch_size, int input_size, int hidden_size);
#endif

void slstm_forward(float* input, float* hidden_state, float* cell_state, float* weights, float* output, int batch_size, int input_size, int hidden_size) {

#ifdef USE_CUDA
    // CUDA implementation
    slstm_forward_cuda(input, hidden_state, cell_state, weights, output, batch_size, input_size, hidden_size);

#else
    // CPU implementation (existing implementation)
    for (int b = 0; b < batch_size; ++b) {
        // Add basic implementation here - for testing purposes.  Replace with the actual implementation
        hidden_state[b] = input[b] + 0.1;
        cell_state[b] = input[b] + 0.2;
        output[b] = input[b] + 0.3;
    }

    std::cout << "Running CPU slstm_forward" << std::endl;
#endif


    std::cout << "Running slstm_forward" << std::endl;
}
