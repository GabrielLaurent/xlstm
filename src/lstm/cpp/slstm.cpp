#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace slstm {

std::vector<float> slstm_forward(const std::vector<float>& input,
                               const std::vector<float>& weights_ih,
                               const std::vector<float>& weights_hh,
                               const std::vector<float>& bias_ih,
                               const std::vector<float>& bias_hh,
                               int input_size,
                               int hidden_size,
                               const std::vector<float>& prev_h,
                               const std::vector<float>& prev_c) {
  // Basic implementation of the sLSTM forward pass.
  // This is a placeholder implementation and will be replaced with optimized version.
  // It assumes that biases can be null.
  int batch_size = 1;
  std::vector<float> output(hidden_size);
  std::vector<float> next_h(hidden_size);
  std::vector<float> next_c(hidden_size);

  // Gates calculations (i, f, g, o)
  std::vector<float> i(hidden_size, 0.0f);
  std::vector<float> f(hidden_size, 0.0f);
  std::vector<float> g(hidden_size, 0.0f);
  std::vector<float> o(hidden_size, 0.0f);

  for (int h = 0; h < hidden_size; ++h) {

    // Input gate
    for (int k = 0; k < input_size; ++k) {
      i[h] += input[k] * weights_ih[h * input_size + k];
    }
    for (int k = 0; k < hidden_size; ++k){
      i[h] += prev_h[k] * weights_hh[h * hidden_size + k];
    }
    if (bias_ih.size() == hidden_size){
      i[h] += bias_ih[h];
    }
    if (bias_hh.size() == hidden_size){
      i[h] += bias_hh[h + hidden_size];
    }
    i[h] = 1.0f / (1.0f + exp(-i[h])); // sigmoid

    // Forget Gate
    for (int k = 0; k < input_size; ++k) {
      f[h] += input[k] * weights_ih[(h + hidden_size) * input_size + k];
    }
    for (int k = 0; k < hidden_size; ++k){
      f[h] += prev_h[k] * weights_hh[(h + hidden_size) * hidden_size + k];
    }
    if (bias_ih.size() == hidden_size){
      f[h] += bias_ih[h + hidden_size];
    }
    if (bias_hh.size() == hidden_size){
      f[h] += bias_hh[h + hidden_size*3];
    }
    f[h] = 1.0f / (1.0f + exp(-f[h])); // sigmoid

    // Cell Gate
    for (int k = 0; k < input_size; ++k) {
      g[h] += input[k] * weights_ih[(h + hidden_size*2) * input_size + k];
    }
    for (int k = 0; k < hidden_size; ++k){
      g[h] += prev_h[k] * weights_hh[(h + hidden_size*2) * hidden_size + k];
    }
    if (bias_ih.size() == hidden_size){
      g[h] += bias_ih[h + hidden_size*2];
    }
    if (bias_hh.size() == hidden_size){
      g[h] += bias_hh[h + hidden_size*4];
    }

    g[h] = tanh(g[h]); // tanh

    //Output Gate
    for (int k = 0; k < input_size; ++k) {
      o[h] += input[k] * weights_ih[(h + hidden_size*3) * input_size + k];
    }
    for (int k = 0; k < hidden_size; ++k){
      o[h] += prev_h[k] * weights_hh[(h + hidden_size*3) * hidden_size + k];
    }
    if (bias_ih.size() == hidden_size){
      o[h] += bias_ih[h + hidden_size*3];
    }
    if (bias_hh.size() == hidden_size){
      o[h] += bias_hh[h + hidden_size*5];
    }
    o[h] = 1.0f / (1.0f + exp(-o[h])); // sigmoid
    
    next_c[h] = f[h] * prev_c[h] + i[h] * g[h];
    next_h[h] = o[h] * tanh(next_c[h]);
    output[h] = next_h[h]; // For now next_h is the output
  }

  return output;
}

PYBIND11_MODULE(slstm_cpp, m) {
  m.doc() = "sLSTM forward pass implemented in C++";
  m.def("forward", &slstm_forward, "Compute the forward pass of the sLSTM block");
}

}
