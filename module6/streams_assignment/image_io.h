// Loading and saving photos (JPG/PNG) with the vendored stb_image
// headers in third_party/.
#pragma once

#include <string>
#include <vector>

// One photo as 8-bit RGBA, row-major, 4 bytes per pixel.
struct Photo {
    std::string name;  // file name without extension
    int width;
    int height;
    std::vector<unsigned char> rgba;
};

// Loads every .jpg/.jpeg/.png in `dir`, sorted by name. Exits with an
// error if there are none or if they are not all the same size.
std::vector<Photo> loadPhotos(const char* dir);

// Writes `rgba` (width x height) to `path` as a JPEG. Creates the parent
// directory if needed.
void saveJpeg(const std::string& path, const unsigned char* rgba, int width,
              int height);
