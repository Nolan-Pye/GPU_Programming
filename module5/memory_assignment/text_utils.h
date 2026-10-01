// Host-side helpers: loading the input text, the byte->letter lookup
// table, the CPU reference histogram, and the letter-frequency chart.
#pragma once

#include <cstddef>
#include "common.h"

// A text buffer in pageable host memory (malloc). Free with free(data).
struct Text {
    unsigned char* data;
    size_t size;
};

// Reads the whole file. If it is a Project Gutenberg eBook, keeps only the
// body between the "*** START OF" and "*** END OF" marker lines.
Text loadText(const char* path);

// table[c] = 0..25 for 'a'/'A'..'z'/'Z', NOT_A_LETTER for any other byte.
void buildLetterTable(signed char table[TABLE_SIZE]);

// Host-memory reference histogram using the same lookup table.
void cpuHistogram(const unsigned char* text, size_t n,
                  const signed char* table, unsigned int hist[NUM_BINS]);

// Prints each letter's count, percentage, and a bar, in two columns.
void printLetterChart(const unsigned int hist[NUM_BINS]);
