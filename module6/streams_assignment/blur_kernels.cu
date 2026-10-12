// Blur kernels for the CUDA Streams and Events assignment. Every variant
// in main.cu launches exactly these two kernels; only the streams, events
// and host memory around them change, so timing differences come from
// scheduling and copy overlap, not from the kernel code.

#include <cmath>
#include "blur_kernels.cuh"

// Every thread reads the same weight at the same time, which is the case
// constant memory's broadcast is built for.
__constant__ float cBlurWeights[BLUR_TAPS];

void buildGaussianWeights(float weights[BLUR_TAPS])
{
    float sum = 0.0f;
    for (int t = -BLUR_RADIUS; t <= BLUR_RADIUS; ++t) {
        float w = expf(-(t * t) / (2.0f * BLUR_SIGMA * BLUR_SIGMA));
        weights[t + BLUR_RADIUS] = w;
        sum += w;
    }
    for (int i = 0; i < BLUR_TAPS; ++i) weights[i] /= sum;
}

void uploadBlurWeights(const float weights[BLUR_TAPS])
{
    CUDA_CHECK(cudaMemcpyToSymbol(cBlurWeights, weights,
                                  BLUR_TAPS * sizeof(float)));
}

// One thread per output pixel; blockIdx.z picks the photo in the group.
// (dx, dy) = (1, 0) blurs along rows, (0, 1) along columns.
template <int DX, int DY>
__global__ void blurKernel(const uchar4* in, uchar4* out, int width,
                           int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    size_t photoOffset = (size_t)blockIdx.z * width * height;
    out[photoOffset + (size_t)y * width + x] =
        blurPixel(in + photoOffset, width, height, x, y, DX, DY,
                  cBlurWeights);
}

static dim3 gridFor(int width, int height, int numPhotos, dim3 block)
{
    return dim3((width + block.x - 1) / block.x,
                (height + block.y - 1) / block.y, numPhotos);
}

void launchBlurHorizontal(const uchar4* in, uchar4* out, int width,
                          int height, int numPhotos, dim3 block,
                          cudaStream_t stream)
{
    blurKernel<1, 0><<<gridFor(width, height, numPhotos, block), block, 0,
                        stream>>>(in, out, width, height);
    CUDA_CHECK(cudaGetLastError());
}

void launchBlurVertical(const uchar4* in, uchar4* out, int width,
                        int height, int numPhotos, dim3 block,
                        cudaStream_t stream)
{
    blurKernel<0, 1><<<gridFor(width, height, numPhotos, block), block, 0,
                        stream>>>(in, out, width, height);
    CUDA_CHECK(cudaGetLastError());
}

void launchBlur(const uchar4* in, uchar4* temp, uchar4* out, int width,
                int height, int numPhotos, dim3 block, cudaStream_t stream)
{
    launchBlurHorizontal(in, temp, width, height, numPhotos, block, stream);
    launchBlurVertical(temp, out, width, height, numPhotos, block, stream);
}
