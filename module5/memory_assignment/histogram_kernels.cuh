// Host-side interface to the letter-histogram kernels. Every kernel runs
// the same per-byte code: look the byte up in a 256-entry table that maps
// it to a letter bin (or NOT_A_LETTER), then atomically add 1 to that bin.
// The kernels differ only in which CUDA memory holds the text, the table,
// and the histogram being updated.
#pragma once

#include <cstddef>
#include "common.h"

enum class Variant {
    HostMapped,  // text read straight from pinned, mapped host memory
    Global,      // text, table, and histogram all in global memory
    Constant,    // lookup table in constant memory
    Shared,      // per-block histogram in shared memory
    Register,    // text loaded 16 bytes at a time into registers
    Combined,    // registers + constant table + shared histogram
};

constexpr Variant ALL_VARIANTS[] = {
    Variant::HostMapped, Variant::Global,   Variant::Constant,
    Variant::Shared,     Variant::Register, Variant::Combined,
};

const char* variantName(Variant v);
const char* variantMemory(Variant v);

// Copies the 256-entry byte->bin table into the __constant__ array.
void uploadConstantTable(const signed char* hostTable);

// Launches one variant. `text` holds n bytes and must be 16-byte aligned;
// `table` is the device (global) copy of the lookup table; `hist` is a
// zeroed NUM_BINS array in global memory that receives the counts.
void launchVariant(Variant v, const unsigned char* text, size_t n,
                   const signed char* table, unsigned int* hist,
                   const LaunchConfig& cfg);
