#pragma once

#include "Common.h"
#include "Math.h"
#include "Thread.h"

#include <vector>

namespace Probulator
{
	template <typename PixelType>
	class ImageBase
	{
	public:

		typedef std::vector<PixelType> PixelArray;
		typedef typename PixelArray::iterator PixelIterator;
		typedef typename PixelArray::const_iterator PixelConstIterator;

		ImageBase()
			: m_size(0, 0)
		{}

		ImageBase(u32 width, u32 height)
			: m_size(width, height)
		{
			m_pixels.resize(m_size.x * m_size.y, PixelType());
		}

		ImageBase(ivec2 size)
			: ImageBase(size.x, size.y)
		{}

		ivec2 getSize() const { return m_size; }
		float getAspect() const { return (float)m_size.x / (float)m_size.y; }

		u32 getWidth() const { return m_size.x; }
		u32 getHeight() const { return m_size.y; }
		u32 getPixelCount() const { return m_size.x * m_size.y; }
		u64	getSizeBytes() const { return m_pixels.size() * sizeof(PixelType); }
		u64 getStrideBytes() const { return m_size.x * sizeof(PixelType); }

		PixelType* getPixels() { return m_pixels.data(); };
		const PixelType* getPixels() const { return m_pixels.data(); };

		PixelIterator begin() { return m_pixels.begin(); }
		PixelIterator end() { return m_pixels.end(); }

		const PixelConstIterator begin() const { return m_pixels.cbegin(); }
		const PixelConstIterator end() const { return m_pixels.cend(); }

		const PixelType& at(u32 x, u32 y) const { return m_pixels[x + m_size.x * y]; }
		PixelType& at(u32 x, u32 y) { return m_pixels[x + m_size.x * y]; }

		const PixelType& at(size_t index) const { return m_pixels[index]; }
		PixelType& at(size_t index) { return m_pixels[index]; }

		const PixelType& at(ivec2 pos) const { return at(pos.x, pos.y); }
		PixelType& at(ivec2 pos) { return at(pos.x, pos.y); }

		void fill(const PixelType& value)
		{
			forPixels([value](PixelType& p){ p = value; });
		}

		// Invoke fun(PixelType& pixel) over all pixels
		template <typename T> void forPixels(T fun)
		{
			for (PixelType& pixel : m_pixels)
			{
				fun(pixel);
			}
		}

		// Invoke fun(PixelType& pixel, u32 index) over all pixels
		template <typename T> void forPixels1D(T fun)
		{
			u32 count = getPixelCount();
			for (u32 i = 0; i < count; ++i)
			{
				fun(at(i), i);
			}
		}

		// Invoke fun(PixelType& pixel, ivec2 position) over all pixels
		template <typename T> void forPixels2D(T fun)
		{
			ivec2 size = getSize();
			for (int y = 0; y < size.y; ++y)
			{
				for (int x = 0; x < size.x; ++x)
				{
					ivec2 pos(x, y);
					fun(at(pos), pos);
				}
			}
		}

		// Invoke fun(PixelType& pixel, ivec2 position) over all pixels in parallel
		template <typename T> inline void parallelForPixels2D(T fun)
		{
			u32 pixelCount = m_size.x * m_size.y;
			parallelFor(0u, pixelCount, [&](u32 i)
			{
				u32 y = i / m_size.x;
				u32 x = i % m_size.x;
				ivec2 pos(x, y);
				fun(at(pos), pos);
			});
		}

	protected:

		ivec2 m_size;
		PixelArray m_pixels;
	};

	class Image : public ImageBase<vec4>
	{
	public:

		Image()
			: ImageBase<vec4>()
		{}

		Image(u32 width, u32 height)
			: ImageBase<vec4>(width, height)
		{}

		Image(ivec2 size)
			: ImageBase<vec4>(size.x, size.y)
		{}

		vec4 sampleNearest(vec2 uv) const;

		float* data() { return m_pixels.empty() ? nullptr : &m_pixels[0].x; }
		const float* data() const { return m_pixels.empty() ? nullptr : &m_pixels[0].x; }

		bool readHdr(const char* filename);

		void writeHdr(const char* filename) const;
		void writePng(const char* filename) const;

		void paste(const Image& src, ivec2 pos);
	};

	Image imageResize(const Image& input, ivec2 newSize);
}
