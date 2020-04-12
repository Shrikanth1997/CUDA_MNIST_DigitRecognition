#include "layer.cu"
#include "kernels.cu"

#include <time.h>

// Define layers of CNN
static Layer_info In_layer = Layer_info(0, 0, 28*28);
static Layer_info L1 = Layer_info(5*5, 6, 24*24*6);
static Layer_info L2 = Layer_info(4*4, 1, 6*6*6);
static Layer_info L3 = Layer_info(6*6*6, 10, 10);

const static float threshold = 1.0E-02f;

static void load_model() {
    FILE *model = fopen("model.dat", "rb");
    if(!model)
        return;
    In_layer.read_model(model);
    L1.read_model(model);
    L2.read_model(model);
    L3.read_model(model);
    fclose(model);
}

static void save_model() {
    FILE *model = fopen("model.dat", "wb");
    In_layer.save_model(model);
    L1.save_model(model);
    L2.save_model(model);
    L3.save_model(model);
    fclose(model);
}

// Forward propagation of a single row in dataset
static double forward_propagation(double data[28][28])
{
    float input[28][28];

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input[i][j] = data[i][j];
        }
    }

    In_layer.clear();
    L1.clear();
    L2.clear();
    L3.clear();

    clock_t start, end;
    start = clock();

    In_layer.setOutput((float *)input);
    
    f_conv_before_act<<<64, 64>>>((float (*)[28])In_layer.opt, (float (*)[24][24])L1.preact, (float (*)[5][5])L1.weight);
    f_conv_bias<<<64, 64>>>((float (*)[24][24])L1.preact, L1.bias);
    call_activation_function<<<64, 64>>>(L1.preact, L1.opt, L1.O);

    f_soft_before_act<<<64, 64>>>((float (*)[24][24])L1.opt, (float (*)[6][6])L2.preact, (float (*)[4][4])L2.weight);
    f_soft_bias<<<64, 64>>>((float (*)[6][6])L2.preact, L2.bias);
    call_activation_function<<<64, 64>>>(L2.preact, L2.opt, L2.O);

    f_final_before_act<<<64, 64>>>((float (*)[6][6])L2.opt, L3.preact, (float (*)[6][6][6])L3.weight);
    f_final_bias<<<64, 64>>>(L3.preact, L3.bias);
    call_activation_function<<<64, 64>>>(L3.preact, L3.opt, L3.O);
    
    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double backward_propagation()
{
    clock_t start, end;

    start = clock();

    b_grad_final<<<64, 64>>>((float (*)[6][6][6])L3.bp_weight, L3.bp_preact, (float (*)[6][6])L2.opt);
    b_bias_final<<<64, 64>>>(L3.bias, L3.bp_preact);

    b_output_soft<<<64, 64>>>((float (*)[6][6])L2.bp_opt, (float (*)[6][6][6])L3.weight, L3.bp_preact);
    b_before_act_soft<<<64, 64>>>((float (*)[6][6])L2.bp_preact, (float (*)[6][6])L2.bp_opt, (float (*)[6][6])L2.preact);
    b_grad_soft<<<64, 64>>>((float (*)[4][4])L2.bp_weight, (float (*)[6][6])L2.bp_preact, (float (*)[24][24])L1.opt);
    b_bias_soft<<<64, 64>>>(L2.bias, (float (*)[6][6])L2.bp_preact);

    b_output_conv<<<64, 64>>>((float (*)[24][24])L1.bp_opt, (float (*)[4][4])L2.weight, (float (*)[6][6])L2.bp_preact);
    b_before_act_conv<<<64, 64>>>((float (*)[24][24])L1.bp_preact, (float (*)[24][24])L1.bp_opt, (float (*)[24][24])L1.preact);
    b_grad_conv<<<64, 64>>>((float (*)[5][5])L1.bp_weight, (float (*)[24][24])L1.bp_preact, (float (*)[28])In_layer.opt);
    b_bias_conv<<<64, 64>>>(L1.bias, (float (*)[24][24])L1.bp_preact);

    apply_gradient<<<64, 64>>>(L3.weight, L3.bp_weight, L3.M * L3.N);
    apply_gradient<<<64, 64>>>(L2.weight, L2.bp_weight, L2.M * L2.N);
    apply_gradient<<<64, 64>>>(L1.weight, L1.bp_weight, L1.M * L1.N);

    end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Returns label of given data (0-9)
static unsigned int classify(double data[28][28], char opt)
{
    float res[10];

    forward_propagation(data);

    unsigned int max = 0;

    cudaMemcpy(res, L3.opt, sizeof(float) * 10, cudaMemcpyDeviceToHost);
    if(opt == 'y')
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                int color = int(255 * data[i][j]);
                fprintf(stdout, "\033[48;2;%d;%d;%dm  \033[0m", color, color, color);
            }
            fprintf(stdout, "\n");
        }
    for (int i = 0; i < 10; ++i) {
        if(opt == 'y') {
            int color = int(res[i]*255);
            fprintf(stdout, " [ %d ] \033[48;2;%d;0;%dm%02.15f\033[0m\n", i, color/4, color, res[i]);
        }
        if (res[max] < res[i]) {
            max = i;
        }
    }

    return max;
}
