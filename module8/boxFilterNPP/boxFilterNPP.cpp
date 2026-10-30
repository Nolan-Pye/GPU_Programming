/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Module 8 practical changes:
//  - NPP 13 (CUDA 13) only ships the stream-context API, so the filter call is
//    nppiFilterBoxBorder_8u_C1R_Ctx with an application-filled NppStreamContext.
//  - Images are read/written as binary PGM directly instead of through
//    ImageIO.h/FreeImage (FreeImage is only vendored for Linux and macOS).
//  - New options: -mask=N (box size), -size=N (synthetic NxN image instead of
//    a file), -iters=N (timed repetitions). A CPU box filter checks the result
//    and gives a speed comparison.
// With no arguments it behaves like the original: Lena.pgm, 5x5 box,
// Lena_boxFilter.pgm.

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdlib>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

// Fill the stream context the _Ctx NPP functions need (NPP no longer does
// this for you). Stream 0 is the legacy default stream.
NppStreamContext makeNppStreamContext(int dev, cudaStream_t stream)
{
    NppStreamContext ctx = {};
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));

    ctx.hStream = stream;
    ctx.nCudaDeviceId = dev;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    checkCudaErrors(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor,
                                           cudaDevAttrComputeCapabilityMajor, dev));
    checkCudaErrors(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor,
                                           cudaDevAttrComputeCapabilityMinor, dev));
    checkCudaErrors(cudaStreamGetFlags(stream, &ctx.nStreamFlags));
    return ctx;
}

// Minimal binary (P5) 8-bit PGM reader/writer, replacing npp::loadImage /
// npp::saveImage from ImageIO.h.
static void skipPgmSpaceAndComments(std::istream &in)
{
    while (in)
    {
        int c = in.peek();
        if (c == '#')
        {
            std::string line;
            std::getline(in, line);
        }
        else if (isspace(c))
        {
            in.get();
        }
        else
        {
            break;
        }
    }
}

void loadPgm(const std::string &rFileName, npp::ImageCPU_8u_C1 &rImage)
{
    std::ifstream in(rFileName.c_str(), std::ios::binary);
    if (!(in.good())) throw npp::Exception("Unable to open " + rFileName, __FILE__, __LINE__);

    std::string magic;
    int width, height, maxval;
    in >> magic;
    if (!(magic == "P5")) throw npp::Exception("Only binary (P5) PGM files are supported", __FILE__, __LINE__);
    skipPgmSpaceAndComments(in); in >> width;
    skipPgmSpaceAndComments(in); in >> height;
    skipPgmSpaceAndComments(in); in >> maxval;
    in.get();   // single whitespace before the pixel data
    if (!(maxval <= 255)) throw npp::Exception("Only 8-bit PGM files are supported", __FILE__, __LINE__);

    npp::ImageCPU_8u_C1 oImage(width, height);
    for (int y = 0; y < height; ++y)
    {
        in.read(reinterpret_cast<char *>(oImage.data() + y * oImage.pitch()), width);
    }
    if (!(in.good())) throw npp::Exception("Truncated PGM file " + rFileName, __FILE__, __LINE__);
    oImage.swap(rImage);
}

void savePgm(const std::string &rFileName, const npp::ImageCPU_8u_C1 &rImage)
{
    std::ofstream out(rFileName.c_str(), std::ios::binary);
    if (!(out.good())) throw npp::Exception("Unable to write " + rFileName, __FILE__, __LINE__);
    out << "P5\n" << rImage.width() << " " << rImage.height() << "\n255\n";
    for (unsigned int y = 0; y < rImage.height(); ++y)
    {
        out.write(reinterpret_cast<const char *>(rImage.data() + y * rImage.pitch()), rImage.width());
    }
}

// Synthetic test image: smooth gradient plus a checkerboard and some noise, so
// a box filter visibly changes it.
void makeSyntheticImage(int n, npp::ImageCPU_8u_C1 &rImage)
{
    npp::ImageCPU_8u_C1 oImage(n, n);
    unsigned int seed = 12345;
    for (int y = 0; y < n; ++y)
    {
        Npp8u *row = oImage.data() + y * oImage.pitch();
        for (int x = 0; x < n; ++x)
        {
            seed = seed * 1664525u + 1013904223u;
            int checker = (((x / 32) + (y / 32)) & 1) ? 64 : 0;
            int v = (x + y) * 128 / (2 * n) + checker + (int)(seed >> 26);
            row[x] = (Npp8u)(v > 255 ? 255 : v);
        }
    }
    oImage.swap(rImage);
}

// Straightforward CPU box filter with replicated borders, same anchor
// convention as the NPP call. The average is truncated, matching NPP.
void cpuBoxFilter(const npp::ImageCPU_8u_C1 &src, npp::ImageCPU_8u_C1 &dst,
                  NppiSize mask, NppiPoint anchor)
{
    const int w = src.width(), h = src.height();
    const int area = mask.width * mask.height;
    for (int y = 0; y < h; ++y)
    {
        Npp8u *out = dst.data() + y * dst.pitch();
        for (int x = 0; x < w; ++x)
        {
            int sum = 0;
            for (int j = 0; j < mask.height; ++j)
            {
                int yy = y - anchor.y + j;
                yy = yy < 0 ? 0 : (yy >= h ? h - 1 : yy);
                const Npp8u *in = src.data() + yy * src.pitch();
                for (int i = 0; i < mask.width; ++i)
                {
                    int xx = x - anchor.x + i;
                    xx = xx < 0 ? 0 : (xx >= w ? w - 1 : xx);
                    sum += in[xx];
                }
            }
            out[x] = (Npp8u)(sum / area);
        }
    }
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        std::string sFilename;
        char *filePath;

        int dev = cudaDeviceInit(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        int maskDim = 5;
        int syntheticSize = 0;
        int iters = 20;
        if (checkCmdLineFlag(argc, (const char **)argv, "mask"))
        {
            maskDim = getCmdLineArgumentInt(argc, (const char **)argv, "mask");
        }
        if (checkCmdLineFlag(argc, (const char **)argv, "size"))
        {
            syntheticSize = getCmdLineArgumentInt(argc, (const char **)argv, "size");
        }
        if (checkCmdLineFlag(argc, (const char **)argv, "iters"))
        {
            iters = getCmdLineArgumentInt(argc, (const char **)argv, "iters");
        }

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        std::string sResultFilename;

        if (syntheticSize > 0)
        {
            makeSyntheticImage(syntheticSize, oHostSrc);
            sResultFilename = "synthetic_boxFilter.pgm";
            std::cout << "boxFilterNPP using synthetic " << syntheticSize << "x"
                      << syntheticSize << " image" << std::endl;
        }
        else
        {
            if (checkCmdLineFlag(argc, (const char **)argv, "input"))
            {
                getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
            }
            else
            {
                filePath = sdkFindFilePath("Lena.pgm", argv[0]);
            }

            if (filePath)
            {
                sFilename = filePath;
            }
            else
            {
                sFilename = "Lena.pgm";
            }

            // if we specify the filename at the command line, then we only test sFilename[0].
            int file_errors = 0;
            std::ifstream infile(sFilename.data(), std::ifstream::in);

            if (infile.good())
            {
                std::cout << "boxFilterNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
                file_errors = 0;
                infile.close();
            }
            else
            {
                std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">" << std::endl;
                file_errors++;
                infile.close();
            }

            if (file_errors > 0)
            {
                exit(EXIT_FAILURE);
            }

            sResultFilename = sFilename;

            std::string::size_type dot = sResultFilename.rfind('.');

            if (dot != std::string::npos)
            {
                sResultFilename = sResultFilename.substr(0, dot);
            }

            sResultFilename += "_boxFilter.pgm";

            // load gray-scale image from disk
            loadPgm(sFilename, oHostSrc);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFilePath);
            sResultFilename = outputFilePath;
        }

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        float uploadMs = 0, filterMs = 0, downloadMs = 0;

        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        checkCudaErrors(cudaEventRecord(start));
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&uploadMs, start, stop));

        // create struct with box-filter mask size
        NppiSize oMaskSize = {maskDim, maskDim};

        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {0, 0};

        // create struct with ROI size
        NppiSize oSizeROI = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
        // allocate device image of appropriately reduced size
        npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
        // set anchor point inside the mask to (oMaskSize.width / 2, oMaskSize.height / 2)
        // It should round down when odd
        NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

        NppStreamContext nppCtx = makeNppStreamContext(dev, 0);

        // run box filter (one untimed warm-up, then the timed repetitions)
        for (int it = 0; it <= iters; ++it)
        {
            if (it == 1)
            {
                checkCudaErrors(cudaEventRecord(start));
            }
            NPP_CHECK_NPP (
                               nppiFilterBoxBorder_8u_C1R_Ctx(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                                              oSrcSize, oSrcOffset,
                                                              oDeviceDst.data(), oDeviceDst.pitch(),
                                                              oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE,
                                                              nppCtx) );
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&filterMs, start, stop));
        filterMs /= (iters > 0 ? iters : 1);

        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        checkCudaErrors(cudaEventRecord(start));
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        checkCudaErrors(cudaEventElapsedTime(&downloadMs, start, stop));

        savePgm(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        double mpix = (double)oSizeROI.width * oSizeROI.height / 1e6;
        printf("image %d x %d (%.2f MPix), mask %d x %d\n",
               oSizeROI.width, oSizeROI.height, mpix, maskDim, maskDim);
        printf("  upload   (H->D) : %8.3f ms\n", uploadMs);
        printf("  NPP box filter  : %8.3f ms  (avg of %d, %.0f MPix/s)\n",
               filterMs, iters, mpix / (filterMs / 1e3));
        printf("  download (D->H) : %8.3f ms\n", downloadMs);

        // CPU reference: correctness check and a speed comparison. Skip when
        // the naive O(W*H*k^2) loop would take too long.
        double cpuWork = (double)oSizeROI.width * oSizeROI.height * maskDim * maskDim;
        if (cpuWork <= 4e9)
        {
            npp::ImageCPU_8u_C1 oHostRef(oDeviceDst.size());
            auto t0 = std::chrono::high_resolution_clock::now();
            cpuBoxFilter(oHostSrc, oHostRef, oMaskSize, oAnchor);
            auto t1 = std::chrono::high_resolution_clock::now();
            double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int maxDiff = 0;
            long long mismatches = 0;
            for (int y = 0; y < oSizeROI.height; ++y)
            {
                for (int x = 0; x < oSizeROI.width; ++x)
                {
                    int d = abs((int)oHostDst.data()[y * oHostDst.pitch() + x] -
                                (int)oHostRef.data()[y * oHostRef.pitch() + x]);
                    if (d > maxDiff) maxDiff = d;
                    if (d) ++mismatches;
                }
            }
            printf("  CPU box filter  : %8.3f ms  (single thread, naive)\n", cpuMs);
            printf("  speedup (kernel only)     : %.1fx\n", cpuMs / filterMs);
            printf("  speedup (incl. transfers) : %.1fx\n",
                   cpuMs / (filterMs + uploadMs + downloadMs));
            printf("  vs CPU: max |diff| = %d, pixels differing = %lld of %d\n",
                   maxDiff, mismatches, oSizeROI.width * oSizeROI.height);
        }
        else
        {
            printf("  CPU box filter  : skipped (W*H*k^2 = %.2g too large)\n", cpuWork);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
