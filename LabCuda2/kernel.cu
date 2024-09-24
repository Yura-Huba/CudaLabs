
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <fstream>

#define BLOCK_SIZE 32
#define FILTER_WIDTH 31
#define FILTER_HEIGHT 31
#define MODE 0

using namespace std;
using namespace cv;

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void boxFilter(unsigned char* imputImage, unsigned char* outputImage, unsigned int width, unsigned int height, unsigned int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int girthByY = FILTER_HEIGHT / 2;
    int girthByX = FILTER_WIDTH / 2;

    if ((x >= girthByX) && (x < (width - girthByX)) && (y >= girthByY) && (y < (height - girthByY)))
    {
        for (int c = 0; c < channels; c++) {
            float sum = 0;

            for (int ky = -girthByY; ky <= girthByY; ky++) {
                for (int kx = -girthByX; kx <= girthByX; kx++) {
                    float pixel = imputImage[((y + ky) * width + (x + kx))*channels + c];
                    sum += pixel;
                }
            }
            outputImage[(y * width + x) * channels + c] = sum / (FILTER_WIDTH * FILTER_HEIGHT);
        }
    }
}

__global__ void boxFilterWithTexture(cudaTextureObject_t textureObj, unsigned char* outputImage, unsigned int width, unsigned int height, unsigned char channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int girthByY = FILTER_HEIGHT / 2;
    int girthByX = FILTER_WIDTH / 2;

    if ((x >= girthByX) && (x < (width - girthByX)) && (y >= girthByY) && (y < (height - girthByY)))
    {
        float sum[] = { 0, 0, 0, 0 };

        for (int ky = -girthByY; ky <= girthByY; ky++) {
            for (int kx = -girthByX; kx <= girthByX; kx++) {
                uchar4 pixel = tex2D<uchar4>(textureObj, x + kx, y + ky);
                sum[0] += pixel.x;
                sum[1] += pixel.y;
                sum[2] += pixel.z;
                sum[3] += pixel.w;
            }
        }

        for (int i = 0; i < channels; i++) {
            outputImage[(y * width + x) * channels + i] = sum[i] / (FILTER_WIDTH * FILTER_HEIGHT);
        }
    }
}

__global__ void boxFilterWithSharedMem(cudaTextureObject_t textureObj, unsigned char* outputImage, unsigned int width, unsigned int height, unsigned char channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int girthByY = FILTER_HEIGHT / 2;
    int girthByX = FILTER_WIDTH / 2;

    __shared__ uchar4 sharedMem[BLOCK_SIZE + FILTER_WIDTH][BLOCK_SIZE + FILTER_HEIGHT];  

    int local_X = threadIdx.x + girthByX;
    int local_Y = threadIdx.y + girthByY;

    uchar4 pixel = tex2D<uchar4>(textureObj, x, y);
    sharedMem[local_X][local_Y] = pixel;

    int x_ = threadIdx.x - girthByX;
    int x__ = threadIdx.x + girthByX;
    int y_ = threadIdx.y - girthByY;
    int y__ = threadIdx.y + girthByY;

    bool xBorderNeg = (x - girthByX >= 0);
    bool xBorderPos = (x + girthByX < width);
    bool yBorderNeg = (y - girthByY >= 0);
    bool yBorderPos = (y + girthByY < height);

    if ((x_ < 0 || y_  < 0) && xBorderNeg && yBorderNeg) {
        pixel = tex2D<uchar4>(textureObj, x - girthByX, y - girthByY);
        sharedMem[local_X - girthByX][local_Y - girthByY] = pixel;
    }

    if (((x__ >= BLOCK_SIZE) || (y__ >= BLOCK_SIZE )) && xBorderPos && yBorderPos) {
        pixel = tex2D<uchar4>(textureObj, x + girthByX, y + girthByY);
        sharedMem[local_X + girthByX][local_Y + girthByY] = pixel;
    }

    if ((x_ < girthByX && y__ >= (BLOCK_SIZE - girthByY)) && (x_ < 0 || y__ >= BLOCK_SIZE) && xBorderNeg && yBorderPos) {
        pixel = tex2D<uchar4>(textureObj, x - girthByX, y + girthByY);
        sharedMem[local_X - girthByX][local_Y + girthByY] = pixel;
    }

    if ((x__ >= (BLOCK_SIZE - girthByX) && y_ < girthByY) && (x__ >=  BLOCK_SIZE || y_ < 0) && xBorderPos && yBorderNeg) {
        pixel = tex2D<uchar4>(textureObj, x + girthByX, y - girthByY);
        sharedMem[local_X + girthByX][local_Y - girthByY] = pixel;
    }

    /*if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 1; i <= girthByX && x - i >= 0; i++) {
            for (int j = 1; j <= girthByY && y - i >= 0; j++) {
                pixel = tex2D<uchar4>(textureObj, x - i, y - j);
                sharedmem[local_X - i][local_Y - j] = pixel;
            }
        }
    }
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
        for (int i = 1; i <= girthByX && x - i >= 0; i++) {
            for (int j = 1; j <= girthByY && y + i < height; j++) {
                pixel = tex2D<uchar4>(textureObj, x - i, y + j);
                sharedmem[local_X - i][local_Y + j] = pixel;
            }
        }
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
        for (int i = 1; i <= girthByX && x + i < width; i++) {
            for (int j = 1; j <= girthByY && y - i >= 0; j++) {
                pixel = tex2D<uchar4>(textureObj, x + i, y - j);
                sharedmem[local_X + i][local_Y - j] = pixel;
            }
        }
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
        for (int i = 1; i <= girthByX && x + i < width; i++) {
            for (int j = 1; j <= girthByY && y + i < height; j++) {
                pixel = tex2D<uchar4>(textureObj, x + i, y + j);
                sharedmem[local_X + i][local_Y + j] = pixel;
            }
        }
    }

    if (threadIdx.x == 0) {
        for (int i = 1; i <= girthByX && x - i >= 0; i++) {
            pixel = tex2D<uchar4>(textureObj, x - i, y);
            sharedmem[local_X - i][local_Y] = pixel;
        }
    }
    if (threadIdx.y == 0) {
        for (int i = 1; i <= girthByY && y - i >= 0; i++) {
            pixel = tex2D<uchar4>(textureObj, x, y - i);
            sharedmem[local_X][local_Y - i] = pixel;
        }
    }
    if (threadIdx.x == BLOCK_SIZE - 1) {
        for (int i = 1; i <= girthByX && x + i < width; i++) {
            pixel = tex2D<uchar4>(textureObj, x + i, y);
            sharedmem[local_X + i][local_Y] = pixel;
        }
    }
    if (threadIdx.y == BLOCK_SIZE - 1) {
        for (int i = 1; i <= girthByY && y + i < height; i++) {
            pixel = tex2D<uchar4>(textureObj, x, y + i);
            sharedmem[local_X][local_Y + i] = pixel;
        }
    }*/

    __syncthreads();

    if ((x >= girthByX) && (x < (width - girthByX)) && (y >= girthByY) && (y < (height - girthByY)))
    {
        float sum[] = {0, 0, 0, 0};

        for (int ky = -girthByY; ky <= girthByY; ky++) {
            for (int kx = -girthByX; kx <= girthByX; kx++) {
                pixel = sharedMem[local_X + kx][local_Y + ky];
                sum[0] += pixel.x;
                sum[1] += pixel.y;
                sum[2] += pixel.z;
                sum[3] += pixel.w;
            }
        }
    
        for (int i = 0; i < channels; i++) {
            outputImage[(y * width + x) * channels + i] = sum[i] / (FILTER_WIDTH * FILTER_HEIGHT);
        }
    }
}

cudaTextureObject_t createTexture(const Mat image) {
    cudaTextureObject_t textureObject = 0;
    cudaArray_t array;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    gpuErrorCheck(cudaMallocArray(&array, &channelDesc, image.cols, image.rows));

    gpuErrorCheck(cudaMemcpy2DToArray(array, 0, 0, image.data, image.step, image.cols * sizeof(uchar4), image.rows, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false; 

    gpuErrorCheck(cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

    return textureObject;
}


extern "C" void filter(const cv::Mat& input, cv::Mat& output)
{
    cudaEvent_t start, stop;
    gpuErrorCheck(cudaEventCreate(&start));
    gpuErrorCheck(cudaEventCreate(&stop));

    const int inputSize = input.cols * input.rows * input.channels();
    const int outputSize = output.cols * output.rows * output.channels();

#if MODE > 0
    cudaTextureObject_t texObject = createTexture(input);
#else
    unsigned char* d_input;
    gpuErrorCheck(cudaMalloc<unsigned char>(&d_input, inputSize));
    gpuErrorCheck(cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice));
#endif
    unsigned char* d_output;

    gpuErrorCheck(cudaMalloc<unsigned char>(&d_output, outputSize));


    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

    gpuErrorCheck(cudaEventRecord(start));

#if MODE == 1
    boxFilterWithTexture << <grid, block >> > (texObject, d_output, output.cols, output.rows, input.channels());
#elif MODE == 2
    boxFilterWithSharedMem << <grid, block >> > (texObject, d_output, output.cols, output.rows, input.channels());
#else
    boxFilter <<<grid, block >>> (d_input, d_output, output.cols, output.rows, input.channels());
#endif

    gpuErrorCheck(cudaEventRecord(stop));

    gpuErrorCheck(cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(d_output));

#if MODE > 0
    gpuErrorCheck(cudaDestroyTextureObject(texObject))
#else
    gpuErrorCheck(cudaFree(d_input));
#endif

    gpuErrorCheck(cudaEventSynchronize(stop));
    float milliseconds = 0;

    gpuErrorCheck(cudaEventElapsedTime(&milliseconds, start, stop));
    cout << "Time: " << milliseconds << "\n";
}

int main()
{
    int deviceCount = 0;

    gpuErrorCheck(cudaGetDeviceCount(&deviceCount));

    int device = 0;
    cudaDeviceProp deviceProperties;
    gpuErrorCheck(cudaGetDeviceProperties(&deviceProperties, device));
    gpuErrorCheck(cudaSetDevice(device));

    string image_name = "Cat";

    string input_file = image_name + ".jpg";
    string output_file_gpu = image_name + "_gpu.jpg";

    Mat srcImage = imread(input_file, IMREAD_UNCHANGED);
    if (srcImage.empty())
    {
        std::cout << "File not found: " << input_file << std::endl;
        return EXIT_FAILURE;
    } 
    cvtColor(srcImage, srcImage, COLOR_BGR2BGRA);

    cout << "Image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    Mat dstImage(srcImage.size(), srcImage.type());

    filter(srcImage, dstImage);

    imwrite(output_file_gpu, dstImage);

    return EXIT_SUCCESS;
}

