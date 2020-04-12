#include <cstdio>

class Layer_info {
    public:
    int M, N, O;

    float *opt;
    float *preact;

    float *bias;
    float *weight;

    float *bp_opt;
    float *bp_preact;
    float *bp_weight;

    Layer_info(int M, int N, int O);

    ~Layer_info();

    void setOutput(float *data);
    void clear();
    void bp_clear();
    void save_model(FILE*);
    void read_model(FILE*);
};

// Constructor
Layer_info::Layer_info(int M, int N, int O)
{
    this->M = M;
    this->N = N;
    this->O = O;

    float h_bias[N];
    float h_weight[N][M];

    opt = NULL;
    preact = NULL;
    bias   = NULL;
    weight = NULL;

    for (int i = 0; i < N; ++i) {
        h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
        /*h_bias[i] = 0.0f;*/

        for (int j = 0; j < M; ++j) {
            h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
            /*h_weight[i][j] = 0.05f;*/
        }
    }

    cudaMalloc(&opt, sizeof(float) * O);
    cudaMalloc(&preact, sizeof(float) * O);

    cudaMalloc(&bias, sizeof(float) * N);

    cudaMalloc(&weight, sizeof(float) * M * N);

    cudaMalloc(&bp_opt, sizeof(float) * O);
    cudaMalloc(&bp_preact, sizeof(float) * O);
    cudaMalloc(&bp_weight, sizeof(float) * M * N);

    cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// Destructor
Layer_info::~Layer_info()
{
    cudaFree(opt);
    cudaFree(preact);

    cudaFree(bias);

    cudaFree(weight);

    cudaFree(bp_opt);
    cudaFree(bp_preact);
    cudaFree(bp_weight);
}

// Send data one row from dataset to the GPU
void Layer_info::setOutput(float *data)
{
    cudaMemcpy(opt, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer_info::clear()
{
    cudaMemset(opt, 0x00, sizeof(float) * O);
    cudaMemset(preact, 0x00, sizeof(float) * O);
}

void Layer_info::bp_clear()
{
    cudaMemset(bp_opt, 0x00, sizeof(float) * O);
    cudaMemset(bp_preact, 0x00, sizeof(float) * O);
    cudaMemset(bp_weight, 0x00, sizeof(float) * M * N);
}

void Layer_info::save_model(FILE *model)
{
    char buffer[100000];
    fwrite((char*)&M, sizeof(int), 1, model);
    fwrite((char*)&N, sizeof(int), 1, model);

    cudaMemcpy(buffer, (char*)bias, sizeof(float) * N, cudaMemcpyDeviceToHost);
    fwrite(buffer, sizeof(float) * N, 1, model);
    cudaMemcpy(buffer, (char*)weight, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    fwrite(buffer, sizeof(float) * M * N, 1, model);
}

void Layer_info::read_model(FILE *model)
{
    char buffer[100000];
    
    fread((char*)&M, sizeof(int), 1, model);
    fread((char*)&N, sizeof(int), 1, model);
    
    fread(buffer, sizeof(float) * N, 1, model);
    cudaMemcpy(bias, (float*)buffer, sizeof(float) * N, cudaMemcpyHostToDevice);
    fread(buffer, sizeof(float) * M * N, 1, model);
    cudaMemcpy(weight, (float*)buffer, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}
