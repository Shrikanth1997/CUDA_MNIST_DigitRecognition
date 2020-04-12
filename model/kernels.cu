
#include <cuda.h>
#include <cublas_v2.h>

const static float learning_rate = 1.0E-01f;

// Kernels rquired for forward propagation

// Activation function is sigmoid
__device__ float activation_function(float x)
{
    return 1 / (1 + exp(-x));
}

// Function applies activation function on the input and stores the result in output
__global__ void call_activation_function(float *input, float *output, const int N)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        output[i] = activation_function(input[i]);
    }
}

// first layer
__global__ void f_conv_before_act(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 5*5*6*24*24;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 5);
        const int i2 = ((i /= 5    ) % 5);
        const int i3 = ((i /= 5    ) % 6);
        const int i4 = ((i /= 6    ) % 24);
        const int i5 = ((i /= 24    ) % 24);

        atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
    }
}

__global__ void f_conv_bias(float preact[6][24][24], float bias[6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*24*24;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 24);
        const int i3 = ((i /= 24    ) % 24);

        preact[i1][i2][i3] += bias[i1];
    }
}

// second layer
__global__ void f_soft_before_act(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 4*4*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 4);
        const int i2 = ((i /= 4    ) % 4);
        const int i3 = ((i /= 4    ) % 6);
        const int i4 = ((i /= 6    ) % 6);
        const int i5 = ((i /= 6    ) % 6);

        atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
    }
}

__global__ void f_soft_bias(float preact[6][6][6], float bias[1])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 6);
        const int i3 = ((i /= 6    ) % 6);

        preact[i1][i2][i3] += bias[0];
    }
}

// third and final layer
__global__ void f_final_before_act(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 10*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 10);
        const int i2 = ((i /= 10    ) % 6);
        const int i3 = ((i /= 6    ) % 6);
        const int i4 = ((i /= 6    ) % 6);

        atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
    }
}

__global__ void f_final_bias(float preact[10], float bias[10])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 10;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        preact[i] += bias[i];
    }
}




// Kernels related to backward propagation

// Calculates the error between the output generated and the actual label from the training data
__global__ void calc_error(float *error, float *output, unsigned int label, const int N)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        error[i] = ((label == i ? 1.0f : 0.0f) - output[i]);
    }
}

// Update the ouput with respect to the calculated gradient
__global__ void apply_gradient(float *output, float *grad, const int N)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        output[i] += learning_rate * grad[i];
    }
}
// We go layer by layer in reverse for backward propagation
// third layer
__global__ void b_grad_final(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 10*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 10);
        const int i2 = ((i /= 10    ) % 6);
        const int i3 = ((i /= 6    ) % 6);
        const int i4 = ((i /= 6    ) % 6);

        d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
    }
}

__global__ void b_bias_final(float bias[10], float d_preact[10])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 10;

    for (int i = N * index / stride; i < N * (index+1) / stride; ++i) {
        bias[i] += learning_rate * d_preact[i];
    }
}

// second layer
__global__ void b_output_soft(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 10*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 10);
        const int i2 = ((i /= 10    ) % 6);
        const int i3 = ((i /= 6    ) % 6);
        const int i4 = ((i /= 6    ) % 6);

        atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
    }
}

__global__ void b_before_act_soft(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 6);
        const int i3 = ((i /= 6    ) % 6);

        const float o = activation_function(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

__global__ void b_grad_soft(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 1*4*4*6*6*6;
    const float d = pow(6.0f, 3.0f);

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 1);
        const int i2 = ((i /= 1    ) % 4);
        const int i3 = ((i /= 4    ) % 4);
        const int i4 = ((i /= 4    ) % 6);
        const int i5 = ((i /= 6    ) % 6);
        const int i6 = ((i /= 6    ) % 6);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
    }
}

__global__ void b_bias_soft(float bias[1], float d_preact[6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*6*6;
    const float d = pow(6.0f, 3.0f);

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 6);
        const int i3 = ((i /= 6    ) % 6);

        atomicAdd(&bias[0], learning_rate * d_preact[i1][i2][i3] / d);
    }
}

// first layer
__global__ void b_output_conv(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 1*4*4*6*6*6;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 1);
        const int i2 = ((i /= 1    ) % 4);
        const int i3 = ((i /= 4    ) % 4);
        const int i4 = ((i /= 4    ) % 6);
        const int i5 = ((i /= 6    ) % 6);
        const int i6 = ((i /= 6    ) % 6);

        atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
    }
}

__global__ void b_before_act_conv(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*24*24;

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 24);
        const int i3 = ((i /= 24    ) % 24);

        const float o = activation_function(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

__global__ void b_grad_conv(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*5*5*24*24;
    const float d = pow(24.0f, 2.0f);

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 5);
        const int i3 = ((i /= 5    ) % 5);
        const int i4 = ((i /= 5    ) % 24);
        const int i5 = ((i /= 24    ) % 24);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
    }
}

__global__ void b_bias_conv(float bias[6], float d_preact[6][24][24])
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int N = 6*24*24;
    const float d = pow(24.0f, 2.0f);

    for (int n = N * index / stride; n < N * (index+1) / stride; ++n) {
        int i = n;
        const int i1 = ((i /= 1    ) % 6);
        const int i2 = ((i /= 6    ) % 24);
        const int i3 = ((i /= 24    ) % 24);

        atomicAdd(&bias[i1], learning_rate * d_preact[i1][i2][i3] / d);
    }
}
