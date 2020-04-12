#include "model/propagation.cu"
#include "load_model.c"

#include <unistd.h> //for sleep in test()

static mnist_img *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Data and model loading methods
static void load_data(int train)
{
    if(train)
        mnist_load_model("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte", &train_set, &train_cnt);
    else 
        mnist_load_model("dataset/t10k-images-idx3-ubyte", "dataset/t10k-labels-idx1-ubyte", &test_set, &test_cnt);
}


static void learn(int iter)
{
    static cublasHandle_t blas;
    cublasCreate(&blas);

    float err;
    
    double time_taken = 0.0;

    fprintf(stdout ,"Learning\n");

    while (iter < 0 || iter-- > 0) {
        err = 0.0f;

        for (int i = 0; i < train_cnt; ++i) {
            float tmp_err;

            time_taken += forward_propagation(train_set[i].image);

            L3.bp_clear();
            L2.bp_clear();
            L1.bp_clear();

            // Euclid distance of train_set[i]
            calc_error<<<10, 1>>>(L3.bp_preact, L3.opt, train_set[i].label, 10);
            cublasSnrm2(blas, 10, L3.bp_preact, 1, &tmp_err);
            err += tmp_err;

            time_taken += backward_propagation();
        }

        err /= train_cnt;
        fprintf(stdout, "error: %e, GPU Time: %lf\n", err, time_taken);

        if (err < threshold) {
            fprintf(stdout, "Training complete, error less than threshold\n\n");
            break;
        }

    }
    
    fprintf(stdout, "\n Time - %lf\n", time_taken);
}

// Perform forward propagation of test data
static void test()
{
    int error = 0, res; char opt;
    fprintf(stdout, "Show images? [y/n]:");
    fscanf(stdin, "%c", &opt);
    for (int i = 0; i < test_cnt; ++i) {
        if(opt == 'y')
            fprintf(stdout, "\033[2J\033[1;1H");
        res = classify(test_set[i].image, opt);
        if (res != test_set[i].label)
            ++error;
        if(opt == 'y') {
            fprintf(stdout, "\033[1;3");
            if (res != test_set[i].label)
                fprintf(stdout, "1");
            else
                fprintf(stdout, "2");
            fprintf(stdout, "m  %f Accuracy\n -------------------\n  ----------------------\033[0m", 100 * ( 1 - error /float(1 + i)));
            sleep(1);
        }
    }
    fprintf(stdout, "%f Accuracy\n", 100 * ( 1 - error /float(test_cnt)));
}

int main(int argc, const  char **argv)
{
    srand(time(NULL));

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
        return 1;
    }
    if (argc == 2) {
        load_data(1);
        load_model();
        learn(atoi(argv[1]));
        save_model();
    } else {
        load_data(0);
        load_model();
        test();
    }

    return 0;
}
