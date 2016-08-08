#include "Image.h"

#include <stb_image_write.h>
#include <stb_image.h>
#include <stb_image_resize.h>

namespace Probulator
{
	void Image::writePng(const char* filename) const
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

	bool Image::readPng(const char* filename)
	{
		int w, h, comp;
		u8* imageData = stbi_load(filename, &w, &h, &comp, 3);
		if (!imageData)
		{
			printf("ERROR: Failed to load image from file '%s'\n", filename);
			return false;
		}

		ivec2 size(w, h);
		*this = Image(size);

		for (size_t i = 0; i < m_pixels.size(); ++i)
		{
			u8 r = imageData[i*3 + 0];
			u8 g = imageData[i*3 + 1];
			u8 b = imageData[i*3 + 2];
			m_pixels[i] = vec4(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
		}

		free(imageData);

		return true;
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

	void Image::writeHdr(const char* filename) const
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

	vec4 Image::sampleCrossNearest(vec3 dir) const
	{
		vec2 offset;
		vec2 faceUV;

		vec3 absDir = abs(dir);
		float maxComponent = max(max(absDir.x, absDir.y), absDir.z);

		if (absDir.x > absDir.y && absDir.x > absDir.z)
		{
			faceUV = vec2(dir.x > 0.0f ? dir.z : -dir.z, dir.y);
			offset = dir.x > 0.0f ? vec2(2.0f, 1.0f) : vec2(0.0f, 1.0f);
		}
		else if (absDir.y > absDir.x && absDir.y > absDir.z)
		{
			faceUV = vec2(dir.x, dir.y > 0.0f ? dir.z : -dir.z);
			offset = dir.y > 0.0f ? vec2(1.0f, 0.0f) : vec2(1.0f, 2.0f);
		}
		else
		{
			faceUV = vec2(dir.x, dir.z > 0.0f ? -dir.y : dir.y);
			offset = dir.z > 0.0f ? vec2(1.0f, 3.0f) : vec2(1.0f, 1.0f);
		}

		faceUV = vec2(
			+0.5f * (faceUV.x / maxComponent) + 0.5f, 
			-0.5f * (faceUV.y / maxComponent) + 0.5f);

		return sampleNearest((offset + faceUV) / vec2(3.0f, 4.0f));
	}

	Image imageConvertCrossToLatLong(const Image& input, ivec2 outputSize)
	{
		Image result(outputSize);

		for (int y = 0; y < outputSize.y; ++y)
		{
			for (int x = 0; x < outputSize.x; ++x)
			{
				vec2 uv = (vec2(x, y) + vec2(0.5f)) / vec2(outputSize);
				vec3 dir = latLongTexcoordToCartesian(uv);
				result.at(x, y) = input.sampleCrossNearest(dir);
			}
		}

		return result;
	}

	Image imageResize(const Image& input, ivec2 newSize)
	{
		Image output(newSize);
		stbir_resize_float(input.data(), input.getWidth(), input.getHeight(), (int)input.getStrideBytes(), 
			output.data(), output.getWidth(), output.getHeight(), (int)output.getStrideBytes(), 4);
		return output;
	}

	Image imageDifference(const Image& reference, const Image& image)
	{
		ivec2 size = min(reference.getSize(), image.getSize());

		Image result(size);

		for (int y = 0; y < size.y; ++y)
		{
			for (int x = 0; x < size.x; ++x)
			{
				vec4 error = reference.at(x, y) - image.at(x, y);
				result.at(x, y) = error;
			}
		}
		
		return result;
	}

	Image imageSymmetricAbsolutePercentageError(const Image& reference, const Image& image, const Image* errorWeight)
	{
		ivec2 size = min(reference.getSize(), image.getSize());

		Image result(size);

		for (int y = 0; y < size.y; ++y)
		{
			for (int x = 0; x < size.x; ++x)
			{
				vec4 absDiff = abs(reference.at(x, y) - image.at(x, y));
				if (errorWeight)
				{
					absDiff *= errorWeight->at(x, y);
				}
				vec4 sum = reference.at(x, y) + image.at(x, y);
				result.at(x, y) = absDiff / sum;
			}
		}

		return result;
	}

	vec4 imageMeanSquareError(const Image& reference, const Image& image, const Image* errorWeight)
	{
		vec4 errorSquaredSum = vec4(0.0f);

		ivec2 size = min(reference.getSize(), image.getSize());
		if (errorWeight)
		{
			size = min(size, errorWeight->getSize());
		}
		for (int y = 0; y < size.y; ++y)
		{
			for (int x = 0; x < size.x; ++x)
			{
				vec4 error = reference.at(x, y) - image.at(x, y);
				if (errorWeight)
				{
					error *= errorWeight->at(x, y);
				}
				errorSquaredSum += error * error;
			}
		}

		errorSquaredSum /= size.x * size.y;

		return errorSquaredSum;
	}
}
