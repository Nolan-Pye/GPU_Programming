// Host-side helpers: text loading, lookup table, CPU reference, and chart.

#include "text_utils.h"

#include <cstring>

// Gutenberg marker lines that surround the actual book text.
static const char* const START_MARKER = "*** START OF";
static const char* const END_MARKER = "*** END OF";
// Widest bar in the letter chart, in characters.
static const int BAR_WIDTH = 20;

static unsigned char* readFile(const char* path, size_t* size)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open text file '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    // +1 for a terminating NUL so the marker search can use strstr().
    unsigned char* data = (unsigned char*)malloc((size_t)len + 1);
    if (!data || fread(data, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "Cannot read text file '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(f);
    data[len] = '\0';
    *size = (size_t)len;
    return data;
}

Text loadText(const char* path)
{
    size_t size = 0;
    unsigned char* data = readFile(path, &size);
    char* text = (char*)data;

    // Body starts on the line after the start marker, ends at the end
    // marker. Without the markers, the whole file is used.
    char* begin = text;
    char* end = text + size;
    char* start = strstr(text, START_MARKER);
    if (start) {
        char* eol = strchr(start, '\n');
        begin = eol ? eol + 1 : start;
    }
    char* stop = strstr(begin, END_MARKER);
    if (stop) {
        end = stop;
    }

    // Move the body to the front of the buffer to keep it 16-byte aligned.
    size_t bodySize = (size_t)(end - begin);
    if (bodySize == 0) {
        fprintf(stderr, "Text file '%s' is empty\n", path);
        exit(EXIT_FAILURE);
    }
    memmove(data, begin, bodySize);
    return Text{data, bodySize};
}

void buildLetterTable(signed char table[TABLE_SIZE])
{
    for (int c = 0; c < TABLE_SIZE; ++c) {
        table[c] = NOT_A_LETTER;
    }
    for (int b = 0; b < NUM_BINS; ++b) {
        table['a' + b] = (signed char)b;
        table['A' + b] = (signed char)b;
    }
}

void cpuHistogram(const unsigned char* text, size_t n,
                  const signed char* table, unsigned int hist[NUM_BINS])
{
    memset(hist, 0, NUM_BINS * sizeof(unsigned int));
    for (size_t i = 0; i < n; ++i) {
        int bin = table[text[i]];
        if (bin != NOT_A_LETTER) {
            ++hist[bin];
        }
    }
}

static void printChartCell(const unsigned int hist[NUM_BINS], int bin,
                           unsigned long long total, unsigned int maxCount)
{
    int bar = (int)((unsigned long long)hist[bin] * BAR_WIDTH / maxCount);
    printf("  %c %8u %5.2f%% %-*.*s", 'a' + bin, hist[bin],
           100.0 * hist[bin] / (double)total, BAR_WIDTH, bar,
           "####################");
}

void printLetterChart(const unsigned int hist[NUM_BINS])
{
    unsigned long long total = 0;
    unsigned int maxCount = 1;
    for (int b = 0; b < NUM_BINS; ++b) {
        total += hist[b];
        if (hist[b] > maxCount) maxCount = hist[b];
    }
    printf("\nLetter frequencies (%llu letters):\n", total);
    if (total == 0) return;

    const int rows = NUM_BINS / 2;  // a-m on the left, n-z on the right
    for (int r = 0; r < rows; ++r) {
        printChartCell(hist, r, total, maxCount);
        printChartCell(hist, r + rows, total, maxCount);
        printf("\n");
    }
}
