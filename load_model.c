#include <cstring>

typedef struct mnist_img {
    double image[28][28]; /* 28x28 data for the image */
    unsigned int label; /* label : 0 to 9 */
} mnist_img;

//Convert the binary value to Unsigned int
static unsigned int mnist_binary_to_int(char *num)
{
    int i;
    unsigned int ret = 0;

    for (i = 0; i < 4; ++i) {
        ret <<= 8;
        ret |= (unsigned char)num[i];
    }

    return ret;
}

/*
 MNIST dataset loader.
 
 Returns 0 if successed.
 
 */
 int mnist_load_model( const char *image_filename, const char *label_filename, mnist_img **img, unsigned int *count)
{
    
    FILE *ifp = fopen(image_filename, "rb");
    FILE *lfp = fopen(label_filename, "rb");

    if (!ifp || !lfp) {
        system("wget \"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\";"
           "wget \"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\";"
           "wget \"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\";"
           "wget \"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\";"
           "gunzip train-images-idx3-ubyte.gz;"
           "gunzip train-labels-idx1-ubyte.gz;"
           "gunzip t10k-images-idx3-ubyte.gz;"
           "gunzip t10k-labels-idx1-ubyte.gz;"
           "mkdir dataset;"
           "mv train-images-idx3-ubyte dataset/;"
           "mv train-labels-idx1-ubyte dataset/;"
           "mv t10k-images-idx3-ubyte dataset/;"
           "mv t10k-labels-idx1-ubyte dataset/" );
        
    }

    int return_code = 0;
    int i;
    char tmp[4];

    unsigned int image_count, label_count;
    unsigned int image_dim[2];

    fread(tmp, 1, 4, ifp);
    if (mnist_binary_to_int(tmp) != 2051) {
        return_code = -2; /* Not a valid image file */
        if (ifp) fclose(ifp);
    }

    fread(tmp, 1, 4, lfp);
    if (mnist_binary_to_int(tmp) != 2049) {
        return_code = -3; /* Not a valid label file */
        if (lfp) fclose(lfp);
    }

    fread(tmp, 1, 4, ifp);
    image_count = mnist_binary_to_int(tmp);

    fread(tmp, 1, 4, lfp);
    label_count = mnist_binary_to_int(tmp);

    if (image_count != label_count) {
        return_code = -4; /* Element counts of 2 files mismatch */
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
    }

    for (i = 0; i < 2; ++i) {
        fread(tmp, 1, 4, ifp);
        image_dim[i] = mnist_binary_to_int(tmp);
    }

    if (image_dim[0] != 28 || image_dim[1] != 28) {
        return_code = -2; /* Not a valid image file */
        if (ifp) fclose(ifp);
        if (lfp) fclose(lfp);
    }

    *count = image_count;
    *img = (mnist_img *)malloc(sizeof(mnist_img) * image_count);

//pragma unroll
    for (i = 0; i < image_count; ++i) {
        int j;
        unsigned char read_data[28 * 28];
        mnist_img *d = &(*img)[i];
        fread(read_data, 1, 28*28, ifp);
        for (j = 0; j < 28*28; ++j) {
            d->image[j/28][j%28] = read_data[j] / 255.0;
        }

        fread(tmp, 1, 1, lfp);
        d->label = tmp[0];
    }

    return return_code;
}
