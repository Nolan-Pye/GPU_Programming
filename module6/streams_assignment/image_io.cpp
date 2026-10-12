// Photo loading/saving built on stb_image and stb_image_write
// (public domain / MIT, vendored in third_party/).

#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

namespace fs = std::filesystem;

constexpr int RGBA_CHANNELS = 4;
constexpr int JPEG_QUALITY = 90;

static bool isImageFile(const fs::path& p)
{
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

std::vector<Photo> loadPhotos(const char* dir)
{
    std::vector<fs::path> files;
    if (fs::is_directory(dir)) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && isImageFile(entry.path())) {
                files.push_back(entry.path());
            }
        }
    }
    if (files.empty()) {
        fprintf(stderr, "No .jpg/.png photos found in %s/\n", dir);
        exit(EXIT_FAILURE);
    }
    std::sort(files.begin(), files.end());

    std::vector<Photo> photos;
    for (const fs::path& file : files) {
        int width, height, channelsInFile;
        unsigned char* pixels = stbi_load(file.string().c_str(), &width,
                                          &height, &channelsInFile,
                                          RGBA_CHANNELS);
        if (!pixels) {
            fprintf(stderr, "Cannot read %s: %s\n", file.string().c_str(),
                    stbi_failure_reason());
            exit(EXIT_FAILURE);
        }
        size_t bytes = (size_t)width * height * RGBA_CHANNELS;
        photos.push_back({file.stem().string(), width, height,
                          std::vector<unsigned char>(pixels,
                                                     pixels + bytes)});
        stbi_image_free(pixels);

        if (width != photos[0].width || height != photos[0].height) {
            fprintf(stderr, "%s is %dx%d; all photos must be %dx%d\n",
                    file.string().c_str(), width, height, photos[0].width,
                    photos[0].height);
            exit(EXIT_FAILURE);
        }
    }
    return photos;
}

void saveJpeg(const std::string& path, const unsigned char* rgba, int width,
              int height)
{
    fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
    if (!stbi_write_jpg(path.c_str(), width, height, RGBA_CHANNELS, rgba,
                        JPEG_QUALITY)) {
        fprintf(stderr, "Cannot write %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
}
