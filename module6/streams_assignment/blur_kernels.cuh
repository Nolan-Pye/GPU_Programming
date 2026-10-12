// The two blur kernels every variant runs, plus the per-pixel math they
// share with the CPU reference.
//
// A Gaussian blur is separable: blurring rows and then columns with a 1-D
// Gaussian gives the same result as one 2-D Gaussian, at a fraction of
// the work. So each photo goes through two kernels:
//
//   Kernel 1 (blurHorizontalKernel): photo -> temp, blur along each row
//   Kernel 2 (blurVerticalKernel):   temp  -> out,  blur along each column
//
// Kernel 2 reads kernel 1's output, so the two must run in order: the
// variants rely on stream ordering for that.
#pragma once

#include "common.h"

// Weighted sum of BLUR_TAPS pixels centred on (x, y), stepping (dx, dy)
// between taps. Pixels past the edge repeat the edge pixel. Shared by the
// GPU kernels and the CPU reference.
__host__ __device__ inline uchar4 blurPixel(const uchar4* image, int width,
                                            int height, int x, int y,
                                            int dx, int dy,
                                            const float* weights)
{
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
    for (int t = -BLUR_RADIUS; t <= BLUR_RADIUS; ++t) {
        int sx = min(max(x + t * dx, 0), width - 1);
        int sy = min(max(y + t * dy, 0), height - 1);
        uchar4 p = image[sy * width + sx];
        float w = weights[t + BLUR_RADIUS];
        r += w * p.x;
        g += w * p.y;
        b += w * p.z;
        a += w * p.w;
    }
    return make_uchar4((unsigned char)(r + 0.5f), (unsigned char)(g + 0.5f),
                       (unsigned char)(b + 0.5f), (unsigned char)(a + 0.5f));
}

// Fills weights[BLUR_TAPS] with a normalised Gaussian (sums to 1).
void buildGaussianWeights(float weights[BLUR_TAPS]);

// Copies the weights into __constant__ memory for the kernels.
void uploadBlurWeights(const float weights[BLUR_TAPS]);

// Launch one pass over `numPhotos` consecutive photos in `stream`, one
// thread per pixel (grid z = photo index).
void launchBlurHorizontal(const uchar4* in, uchar4* out, int width,
                          int height, int numPhotos, dim3 block,
                          cudaStream_t stream);
void launchBlurVertical(const uchar4* in, uchar4* out, int width,
                        int height, int numPhotos, dim3 block,
                        cudaStream_t stream);

// Kernel 1 then kernel 2 on the same photos, both in `stream`.
void launchBlur(const uchar4* in, uchar4* temp, uchar4* out, int width,
                int height, int numPhotos, dim3 block, cudaStream_t stream);
