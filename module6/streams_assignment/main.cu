// CUDA Streams and Events assignment — Module 6
//
// Blurs a batch of photos with a two-pass Gaussian blur (a horizontal
// kernel, then a vertical kernel). Each run copies the batch to the GPU,
// blurs it, and copies it back, scheduled several different ways:
//
//   sync, pageable       cudaMemcpy + default stream, no overlap at all
//   N streams, pinned    batch split into N groups of photos, each group's
//                        copy-in, kernels and copy-out issued into its own
//                        stream, so one group's copies overlap another's
//                        kernels
//   3-stream pipeline    dedicated copy-in / blur / copy-out streams,
//                        chained per photo with cudaStreamWaitEvent
//
// The kernel code is identical in every variant. Events time every run
// (plus a per-stage breakdown), and every variant's output must match the
// synchronous baseline bit for bit. Blurred photos are saved to output/.
//
// Usage: streams_assignment [blockX] [blockY] [batchSize]

#include <algorithm>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "blur_kernels.cuh"
#include "common.h"
#include "image_io.h"

// All host and device buffers for one batch. Photo i of the batch is
// photo (i % number of source photos), so a few photos fill any batch.
struct Buffers {
    int width;
    int height;
    int batchSize;
    size_t photoPixels;  // width * height
    uchar4* pageableIn;  // malloc'd host memory
    uchar4* pageableOut;
    uchar4* pinnedIn;    // page-locked host memory (needed for async copies)
    uchar4* pinnedOut;
    uchar4* devIn;
    uchar4* devTemp;     // horizontal pass output, vertical pass input
    uchar4* devOut;
};

// Streams and events created once and reused by every timed run.
struct StreamSet {
    cudaStream_t groupStreams[MAX_STREAMS];
    // 3-stream pipeline: one stream per stage, linked by per-photo events.
    cudaStream_t copyInStream;
    cudaStream_t blurStream;
    cudaStream_t copyOutStream;
    std::vector<cudaEvent_t> photoUploaded;  // photo is on the device
    std::vector<cudaEvent_t> photoBlurred;   // photo's kernels finished
};

// A run of consecutive photos in the batch.
struct PhotoGroup {
    int first;
    int count;
};

// One way of scheduling the same work, plus where it leaves its output.
struct Variant {
    std::string name;
    std::function<void()> run;
    uchar4* output;
};

// Parses [blockX] [blockY] [batchSize].
static LaunchConfig parseArgs(int argc, char** argv)
{
    LaunchConfig cfg{DEFAULT_BLOCK_X, DEFAULT_BLOCK_Y, DEFAULT_BATCH_SIZE};
    if (argc >= 2) cfg.blockX = atoi(argv[1]);
    if (argc >= 3) cfg.blockY = atoi(argv[2]);
    if (argc >= 4) cfg.batchSize = atoi(argv[3]);

    if (cfg.blockX < 1 || cfg.blockY < 1 ||
        cfg.blockX * cfg.blockY > MAX_BLOCK_THREADS ||
        cfg.batchSize < MAX_STREAMS) {
        fprintf(stderr, "Usage: %s [blockX] [blockY] [batchSize >= %d]\n"
                "  blockX * blockY must be at most %d\n", argv[0],
                MAX_STREAMS, MAX_BLOCK_THREADS);
        exit(EXIT_FAILURE);
    }
    return cfg;
}

// Group `index` of `numGroups` near-equal groups of the batch.
static PhotoGroup groupOf(int batchSize, int numGroups, int index)
{
    int perGroup = (batchSize + numGroups - 1) / numGroups;
    int first = std::min(perGroup * index, batchSize);
    return {first, std::min(perGroup, batchSize - first)};
}

static dim3 blockOf(const LaunchConfig& cfg)
{
    return dim3(cfg.blockX, cfg.blockY);
}

// The three steps for one group of consecutive photos, each issued into
// the given stream: copy in, both blur kernels, copy out.
static void uploadGroup(Buffers& b, PhotoGroup g, cudaStream_t stream)
{
    size_t offset = g.first * b.photoPixels;
    CUDA_CHECK(cudaMemcpyAsync(b.devIn + offset, b.pinnedIn + offset,
                               g.count * b.photoPixels * sizeof(uchar4),
                               cudaMemcpyHostToDevice, stream));
}

static void blurGroup(Buffers& b, const LaunchConfig& cfg, PhotoGroup g,
                      cudaStream_t stream)
{
    size_t offset = g.first * b.photoPixels;
    launchBlur(b.devIn + offset, b.devTemp + offset, b.devOut + offset,
               b.width, b.height, g.count, blockOf(cfg), stream);
}

static void downloadGroup(Buffers& b, PhotoGroup g, cudaStream_t stream)
{
    size_t offset = g.first * b.photoPixels;
    CUDA_CHECK(cudaMemcpyAsync(b.pinnedOut + offset, b.devOut + offset,
                               g.count * b.photoPixels * sizeof(uchar4),
                               cudaMemcpyDeviceToHost, stream));
}

// ---- Variants ---------------------------------------------------------

// Baseline: blocking copies from pageable memory, kernels on the default
// stream. Nothing overlaps.
static void runSyncPageable(Buffers& b, const LaunchConfig& cfg)
{
    size_t bytes = b.batchSize * b.photoPixels * sizeof(uchar4);
    CUDA_CHECK(cudaMemcpy(b.devIn, b.pageableIn, bytes,
                          cudaMemcpyHostToDevice));
    launchBlur(b.devIn, b.devTemp, b.devOut, b.width, b.height, b.batchSize,
               blockOf(cfg), 0);
    CUDA_CHECK(cudaMemcpy(b.pageableOut, b.devOut, bytes,
                          cudaMemcpyDeviceToHost));
}

// Splits the batch into numStreams groups and issues each group's
// copy-in, kernels and copy-out into its own stream. Stream order keeps
// each group's steps in sequence; different groups are free to overlap.
static void runGroupedStreams(Buffers& b, const LaunchConfig& cfg,
                              StreamSet& s, int numStreams)
{
    for (int i = 0; i < numStreams; ++i) {
        PhotoGroup g = groupOf(b.batchSize, numStreams, i);
        cudaStream_t stream = s.groupStreams[i];
        uploadGroup(b, g, stream);
        blurGroup(b, cfg, g, stream);
        downloadGroup(b, g, stream);
    }
}

// One stream per stage, one photo at a time. Events carry the
// dependencies across streams: the blur waits for that photo's upload,
// the download waits for its blur. Meanwhile each stream moves straight
// on to the next photo, so copies and kernels overlap.
static void runEventPipeline(Buffers& b, const LaunchConfig& cfg,
                             StreamSet& s)
{
    for (int p = 0; p < b.batchSize; ++p) {
        PhotoGroup photo{p, 1};

        uploadGroup(b, photo, s.copyInStream);
        CUDA_CHECK(cudaEventRecord(s.photoUploaded[p], s.copyInStream));

        CUDA_CHECK(cudaStreamWaitEvent(s.blurStream, s.photoUploaded[p], 0));
        blurGroup(b, cfg, photo, s.blurStream);
        CUDA_CHECK(cudaEventRecord(s.photoBlurred[p], s.blurStream));

        CUDA_CHECK(cudaStreamWaitEvent(s.copyOutStream, s.photoBlurred[p],
                                       0));
        downloadGroup(b, photo, s.copyOutStream);
    }
}

// ---- Timing -----------------------------------------------------------

// Times one run of `work` with events on the legacy default stream. That
// stream waits for (and is waited on by) every blocking stream, so the
// start/stop events bracket the work in all of them.
static float timeOnceMs(const std::function<void()>& work, cudaEvent_t start,
                        cudaEvent_t stop)
{
    CUDA_CHECK(cudaEventRecord(start, 0));
    work();
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// Per-stage times for the whole batch in a single stream, measured by
// recording an event between every operation.
struct StageTimes {
    float copyIn, horizontal, vertical, copyOut;
    float total() const { return copyIn + horizontal + vertical + copyOut; }
    float blur() const { return horizontal + vertical; }
};

static StageTimes measureStages(Buffers& b, const LaunchConfig& cfg,
                                cudaStream_t stream)
{
    constexpr int NUM_MARKS = 5;
    cudaEvent_t mark[NUM_MARKS];
    for (cudaEvent_t& e : mark) CUDA_CHECK(cudaEventCreate(&e));

    PhotoGroup all{0, b.batchSize};
    CUDA_CHECK(cudaEventRecord(mark[0], stream));
    uploadGroup(b, all, stream);
    CUDA_CHECK(cudaEventRecord(mark[1], stream));
    launchBlurHorizontal(b.devIn, b.devTemp, b.width, b.height, b.batchSize,
                         blockOf(cfg), stream);
    CUDA_CHECK(cudaEventRecord(mark[2], stream));
    launchBlurVertical(b.devTemp, b.devOut, b.width, b.height, b.batchSize,
                       blockOf(cfg), stream);
    CUDA_CHECK(cudaEventRecord(mark[3], stream));
    downloadGroup(b, all, stream);
    CUDA_CHECK(cudaEventRecord(mark[4], stream));
    CUDA_CHECK(cudaEventSynchronize(mark[4]));

    float ms[NUM_MARKS - 1];
    for (int i = 0; i < NUM_MARKS - 1; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], mark[i], mark[i + 1]));
    }
    for (cudaEvent_t& e : mark) CUDA_CHECK(cudaEventDestroy(e));
    return {ms[0], ms[1], ms[2], ms[3]};
}

// ---- Verification -----------------------------------------------------

static bool channelsClose(uchar4 a, uchar4 b)
{
    return abs(a.x - b.x) <= VERIFY_TOLERANCE &&
           abs(a.y - b.y) <= VERIFY_TOLERANCE &&
           abs(a.z - b.z) <= VERIFY_TOLERANCE &&
           abs(a.w - b.w) <= VERIFY_TOLERANCE;
}

// Blurs one photo on the CPU with the same per-pixel code and compares
// every pixel with the GPU's result.
static bool matchesCpu(const uchar4* photo, const uchar4* gpuBlurred,
                       int width, int height, const float* weights)
{
    std::vector<uchar4> temp((size_t)width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            temp[(size_t)y * width + x] =
                blurPixel(photo, width, height, x, y, 1, 0, weights);
        }
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uchar4 expected =
                blurPixel(temp.data(), width, height, x, y, 0, 1, weights);
            if (!channelsClose(expected, gpuBlurred[(size_t)y * width + x])) {
                fprintf(stderr, "CPU mismatch at pixel (%d, %d)\n", x, y);
                return false;
            }
        }
    }
    return true;
}

// ---- Setup / teardown -------------------------------------------------

static void allocBuffers(Buffers& b, const std::vector<Photo>& photos,
                         int batchSize)
{
    b.width = photos[0].width;
    b.height = photos[0].height;
    b.batchSize = batchSize;
    b.photoPixels = (size_t)b.width * b.height;
    size_t photoBytes = b.photoPixels * sizeof(uchar4);
    size_t bytes = batchSize * photoBytes;

    b.pageableIn = (uchar4*)malloc(bytes);
    b.pageableOut = (uchar4*)malloc(bytes);
    if (!b.pageableIn || !b.pageableOut) {
        fprintf(stderr, "Out of host memory\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMallocHost((void**)&b.pinnedIn, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&b.pinnedOut, bytes));
    CUDA_CHECK(cudaMalloc((void**)&b.devIn, bytes));
    CUDA_CHECK(cudaMalloc((void**)&b.devTemp, bytes));
    CUDA_CHECK(cudaMalloc((void**)&b.devOut, bytes));

    for (int i = 0; i < batchSize; ++i) {
        const Photo& src = photos[i % photos.size()];
        memcpy(b.pageableIn + i * b.photoPixels, src.rgba.data(), photoBytes);
    }
    memcpy(b.pinnedIn, b.pageableIn, bytes);
}

static void freeBuffers(Buffers& b)
{
    free(b.pageableIn);
    free(b.pageableOut);
    CUDA_CHECK(cudaFreeHost(b.pinnedIn));
    CUDA_CHECK(cudaFreeHost(b.pinnedOut));
    CUDA_CHECK(cudaFree(b.devIn));
    CUDA_CHECK(cudaFree(b.devTemp));
    CUDA_CHECK(cudaFree(b.devOut));
}

static void createStreams(StreamSet& s, int batchSize)
{
    for (cudaStream_t& st : s.groupStreams) {
        CUDA_CHECK(cudaStreamCreate(&st));
    }
    CUDA_CHECK(cudaStreamCreate(&s.copyInStream));
    CUDA_CHECK(cudaStreamCreate(&s.blurStream));
    CUDA_CHECK(cudaStreamCreate(&s.copyOutStream));
    // Dependency-only events: skipping timestamps makes them cheaper.
    s.photoUploaded.resize(batchSize);
    s.photoBlurred.resize(batchSize);
    for (int p = 0; p < batchSize; ++p) {
        CUDA_CHECK(cudaEventCreateWithFlags(&s.photoUploaded[p],
                                            cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.photoBlurred[p],
                                            cudaEventDisableTiming));
    }
}

static void destroyStreams(StreamSet& s)
{
    for (cudaStream_t& st : s.groupStreams) {
        CUDA_CHECK(cudaStreamDestroy(st));
    }
    CUDA_CHECK(cudaStreamDestroy(s.copyInStream));
    CUDA_CHECK(cudaStreamDestroy(s.blurStream));
    CUDA_CHECK(cudaStreamDestroy(s.copyOutStream));
    for (size_t p = 0; p < s.photoUploaded.size(); ++p) {
        CUDA_CHECK(cudaEventDestroy(s.photoUploaded[p]));
        CUDA_CHECK(cudaEventDestroy(s.photoBlurred[p]));
    }
}

// ---- Reporting --------------------------------------------------------

static void printDeviceInfo(const LaunchConfig& cfg, const Buffers& b,
                            size_t numPhotos)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d, %d SMs, %d copy engine(s), "
           "concurrent kernels: %s)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.asyncEngineCount,
           prop.concurrentKernels ? "yes" : "no");
    printf("Batch: %d photos of %dx%d (%zu distinct, %.1f MB per "
           "direction)   Blur: %d taps per pass\n", b.batchSize, b.width,
           b.height, numPhotos,
           b.batchSize * b.photoPixels * sizeof(uchar4) / 1e6, BLUR_TAPS);
    printf("Block: %dx%d (%d threads)   Threads per kernel: %zu "
           "(one per pixel)\n", cfg.blockX, cfg.blockY,
           cfg.blockX * cfg.blockY, b.batchSize * b.photoPixels);
}

static void printStages(const StageTimes& t)
{
    printf("\nStage breakdown, 1 stream, events between stages (ms):\n");
    printf("  copy in %.3f | kernel 1 (horizontal blur) %.3f | kernel 2 "
           "(vertical blur) %.3f | copy out %.3f\n", t.copyIn, t.horizontal,
           t.vertical, t.copyOut);
    float overlapBound = std::max({t.copyIn, t.blur(), t.copyOut});
    printf("  sum %.3f ms; with perfect copy/compute overlap the longest "
           "stage, %.3f ms, is the floor\n", t.total(), overlapBound);
}

// Saves the baseline's blurred version of each distinct photo.
static void saveBlurredPhotos(const std::vector<Photo>& photos,
                              const Buffers& b)
{
    for (size_t i = 0; i < photos.size(); ++i) {
        std::string path = std::string(OUTPUT_DIR) + "/" + photos[i].name +
                           "_blurred.jpg";
        saveJpeg(path, (const unsigned char*)(b.pageableOut +
                                              i * b.photoPixels),
                 b.width, b.height);
    }
    printf("\nSaved %zu blurred photos to %s/\n", photos.size(), OUTPUT_DIR);
}

// ---- Main -------------------------------------------------------------

int main(int argc, char** argv)
{
    LaunchConfig cfg = parseArgs(argc, argv);
    std::vector<Photo> photos = loadPhotos(PHOTO_DIR);

    float weights[BLUR_TAPS];
    buildGaussianWeights(weights);
    uploadBlurWeights(weights);

    Buffers b;
    allocBuffers(b, photos, cfg.batchSize);
    printDeviceInfo(cfg, b, photos.size());
    StreamSet streams;
    createStreams(streams, cfg.batchSize);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // The first variant is the baseline: every other output must match it
    // bit for bit (same kernels, same inputs), and its time sets "speedup".
    std::vector<Variant> variants;
    variants.push_back({"sync, pageable", [&] { runSyncPageable(b, cfg); },
                        b.pageableOut});
    for (int numStreams : STREAM_COUNTS) {
        std::string name = std::to_string(numStreams) +
                           (numStreams == 1 ? " stream, pinned"
                                            : " streams, pinned");
        variants.push_back({name, [&, numStreams] {
                                runGroupedStreams(b, cfg, streams,
                                                  numStreams);
                            },
                            b.pinnedOut});
    }
    variants.push_back({"3-stream event pipeline",
                        [&] { runEventPipeline(b, cfg, streams); },
                        b.pinnedOut});

    printf("\nEnd-to-end time (copy in + 2 blur kernels + copy out), %d "
           "timed runs after 1 warm-up:\n", NUM_TIMED_RUNS);
    printf("  %-24s", "variant");
    for (int r = 1; r <= NUM_TIMED_RUNS; ++r) printf("   run %d", r);
    printf("  %8s %8s  %s\n", "avg ms", "speedup", "check");

    bool allPass = true;
    float baselineMs = 0.0f;
    size_t bytes = b.batchSize * b.photoPixels * sizeof(uchar4);
    for (size_t v = 0; v < variants.size(); ++v) {
        Variant& var = variants[v];
        memset(var.output, 0, bytes);  // a stale result can't pass
        var.run();                     // warm-up
        CUDA_CHECK(cudaDeviceSynchronize());

        float runMs[NUM_TIMED_RUNS];
        float sumMs = 0.0f;
        for (int r = 0; r < NUM_TIMED_RUNS; ++r) {
            runMs[r] = timeOnceMs(var.run, start, stop);
            sumMs += runMs[r];
        }
        float avgMs = sumMs / NUM_TIMED_RUNS;

        // Baseline: CPU-check each distinct photo. Others: exact match.
        bool pass = true;
        if (v == 0) {
            for (size_t p = 0; p < photos.size(); ++p) {
                size_t offset = p * b.photoPixels;
                pass = pass && matchesCpu(b.pageableIn + offset,
                                          var.output + offset, b.width,
                                          b.height, weights);
            }
        } else {
            pass = memcmp(var.output, variants[0].output, bytes) == 0;
        }
        if (v == 0) baselineMs = avgMs;
        allPass = allPass && pass;

        printf("  %-24s", var.name.c_str());
        for (float ms : runMs) printf(" %7.3f", ms);
        printf("  %8.3f %7.2fx  %s\n", avgMs, baselineMs / avgMs,
               pass ? "PASS" : "FAIL");
    }

    printStages(measureStages(b, cfg, streams.groupStreams[0]));
    saveBlurredPhotos(photos, b);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    destroyStreams(streams);
    freeBuffers(b);
    printf("\n%s\n", allPass
           ? "All variants produce identical output (baseline checked "
             "against the CPU)."
           : "ERROR: some variants do not match.");
    return allPass ? EXIT_SUCCESS : EXIT_FAILURE;
}
