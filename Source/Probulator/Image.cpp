#include "Image.h"

#include <malloc.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace Probulator
{
	void Image::writePng(const char* filename)
	{
		if (m_pixels.empty()) return;
		std::vector<u32> imageLdr(m_size.x * m_size.y);
		for (size_t i = 0; i < imageLdr.size(); ++i)
		{
			u8 r = u8(saturate(at(i).x) * 255.0f);
			u8 g = u8(saturate(at(i).y) * 255.0f);
			u8 b = u8(saturate(at(i).z) * 255.0f);

			imageLdr[i] = r | (g << 8) | (b << 16) | 0xFF000000;
		}

		stbi_write_png(filename, m_size.x, m_size.y, 4, imageLdr.data(), m_size.x * 4);
	}

	void Image::paste(const Image& src, ivec2 pos)
	{
		ivec2 min = ivec2(0);
		ivec2 max = src.getSize();

		for (int y = min.y; y != max.y; ++y)
		{
			for (int x = min.x; x != max.x; ++x)
			{
				ivec2 srcPos = ivec2(x, y);
				ivec2 dstPos = srcPos + pos;
				at(dstPos) = src.at(srcPos);
			}
		}
	}

	bool Image::readHdr(const char* filename)
	{
		int w, h, comp;
		float* imageData = stbi_loadf(filename, &w, &h, &comp, 3);
		if (!imageData)
		{
			printf("ERROR: Failed to load image from file '%s'\n", filename);
			return false;
		}

		ivec2 size(w, h);
		*this = Image(size);

		vec3* inputPixels = reinterpret_cast<vec3*>(imageData);
		for (size_t i = 0; i < m_pixels.size(); ++i)
		{
			m_pixels[i] = vec4(inputPixels[i], 1.0f);
		}

		free(imageData);

		return true;
	}

	void Image::writeHdr(const char* filename)
	{
		if (m_pixels.empty()) return;
		stbi_write_hdr(filename, m_size.x, m_size.y, 4, data());
	}

	vec4 Image::sampleNearest(vec2 uv) const
	{
		ivec2 pos = floor(uv * (vec2)m_size);
		pos = clamp(pos, ivec2(0), m_size - 1);
		return at(pos);
	}

}
