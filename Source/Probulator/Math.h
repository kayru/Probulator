#pragma once

#include "Common.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace Probulator
{
	static const float pi = glm::pi<float>();
	static const float twoPi = 2.0f * glm::pi<float>();
	static const float fourPi = 4.0f * glm::pi<float>();

	using glm::vec2;
	using glm::vec3;
	using glm::vec4;

	using glm::ivec2;
	using glm::ivec3;
	using glm::ivec4;

	using glm::mat2;
	using glm::mat3;
	using glm::mat4;

	using glm::abs;
	using glm::atan;
	using glm::cos;
	using glm::exp;
	using glm::log;
	using glm::max;
	using glm::min;
	using glm::mix;
	using glm::sin;
	using glm::sinh;
	using glm::sqrt;

	template <typename T>
	inline T sqr(const T& x)
	{
		return x*x;
	}

	template <typename T>
	inline T cube(const T& x)
	{
		return x*x*x;
	}

	inline float saturate(float x)
	{
		return max(0.0f, min(x, 1.0f));
	}

	template <typename T>
	inline float dotMax0(const T& a, const T& b)
	{
		return max(0.0f, dot(a, b));
	}

	template <typename T>
	inline float dotSaturate(const T& a, const T& b)
	{
		return saturate(dot(a, b));
	}

	inline void sinCos(float x, float* outSinX, float* outCosX)
	{
		*outSinX = sin(x);
		*outCosX = cos(x);
	}

	inline mat3 makeOrthogonalBasis(vec3 n)
	{
		// http://orbit.dtu.dk/files/57573287/onb_frisvad_jgt2012.pdf

		vec3 b1, b2;

		if (n.z < -0.9999999f)
		{
			b1 = vec3(0.0f, -1.0f, 0.0f);
			b2 = vec3(-1.0f, 0.0f, 0.0f);
		}
		else
		{
			const float a = 1.0f / (1.0f + n.z);
			const float b = -n.x*n.y*a;
			b1 = vec3(1.0f - n.x*n.x*a, b, -n.x);
			b2 = vec3(b, 1.0f - n.y*n.y*a, -n.y);
		}
		
		return mat3(b1, b2, n);
	}

	inline float latLongTexelArea(ivec2 pos, ivec2 imageSize)
	{
		vec2 uv0 = vec2(pos) / vec2(imageSize);
		vec2 uv1 = vec2(pos + 1) / vec2(imageSize);

		float theta0 = pi*(uv0.x*2.0f - 1.0f);
		float theta1 = pi*(uv1.x*2.0f - 1.0f);

		float phi0 = pi*(uv0.y - 0.5f);
		float phi1 = pi*(uv1.y - 0.5f);

		return abs(theta1 - theta0) * abs(sin(phi1) - sin(phi0));
	}


	inline vec2 cartesianToLatLongTexcoord(vec3 p)
	{
		// http://gl.ict.usc.edu/Data/HighResProbes

		float u = (1.0f + atan(p.x, -p.z) / pi);
		float v = acos(p.y) / pi;

		return vec2(u * 0.5f, v);
	}

	inline vec3 latLongTexcoordToCartesian(vec2 uv)
	{
		// http://gl.ict.usc.edu/Data/HighResProbes

		float theta = pi*(uv.x*2.0f - 1.0f);
		float phi = pi*uv.y;

		float x = sin(phi)*sin(theta);
		float y = cos(phi);
		float z = -sin(phi)*cos(theta);

		return vec3(x, y, z);
	}

	inline vec3 sphericalToCartesian(vec2 thetaPhi)
	{
		// https://graphics.stanford.edu/papers/envmap/envmap.pdf

		float theta = thetaPhi.x;
		float phi = thetaPhi.y;

		float sinTheta, cosTheta;
		sinCos(theta, &sinTheta, &cosTheta);

		float sinPhi, cosPhi;
		sinCos(phi, &sinPhi, &cosPhi);

		float x = sinTheta * cosPhi;
		float y = sinTheta * sinPhi;
		float z = cosTheta;

		return vec3(x, y, z);
	}

	inline vec2 cartesianToSpherical(vec3 p)
	{
		// https://graphics.stanford.edu/papers/envmap/envmap.pdf

		float phi = atan(p.y, p.x);
		float theta = acos(p.z);

		return vec2(theta, phi);
	}


	inline vec2 sampleHammersley(u32 i, u32 n)
	{
		u32 bits = i;
		bits = (bits << 16u) | (bits >> 16u);
		bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
		bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
		bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
		bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
		float vdc = float(bits) * 2.3283064365386963e-10f;
		return vec2(float(i) / float(n), vdc);
	}
    
    inline float sampleHalton(u32 index, u32 base)
    {
        float f = 1.f;
        float r = 0.f;
        
        while (index > 0) {
            f = f / float(base);
            r += f * float(index % base);
            index /= base;
        }
        return r;
    }

	inline vec3 sampleUniformHemisphere(float u, float v)
	{
		float phi = v * twoPi;
		float cosTheta = u;
		float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

		float sinPhi, cosPhi;
		sinCos(phi, &sinPhi, &cosPhi);

		return vec3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
	}

	inline vec3 sampleVogelsSphere(u32 i, u32 n)
	{
		// http://blog.marmakoide.org/?p=1

		float goldenAngle = pi * (3.0f - sqrt(5.0f));

		float theta = goldenAngle * i;
		float t = n > 1 ? float(i) / (n - 1) : 0.5f;
		float z = mix(1.0f - 1.0f / n, 1.0f / n - 1.0f, t);
		float radius = sqrt(1.0f - z*z);

		float sinTheta, cosTheta;
		sinCos(theta, &sinTheta, &cosTheta);

		float x = radius * cosTheta;
		float y = radius * sinTheta;

		return vec3(x, y, z);
	}

	inline vec3 sampleUniformSphere(float u, float v)
	{
		float z = 1.0f - 2.0f * u;
		float r = sqrt(max(0.0f, 1.0f - z*z));
		float phi = twoPi * v;

		float sinPhi, cosPhi;
		sinCos(phi, &sinPhi, &cosPhi);

		float x = r * cosPhi;
		float y = r * sinPhi;

		return vec3(x, y, z);
	}

	inline vec3 sampleUniformSphere(vec2 uv)
	{
		return sampleUniformSphere(uv.x, uv.y);
	}

	inline vec3 sampleCosineHemisphere(float u, float v)
	{
		float phi = v * twoPi;
		float cosTheta = sqrt(u);
		float sinTheta = sqrt(1.0f - u);

		float sinPhi, cosPhi;
		sinCos(phi, &sinPhi, &cosPhi);

		float x = cosPhi * sinTheta;
		float y = sinPhi * sinTheta;
		float z = cosTheta;

		return vec3(x, y, z);
	}

	inline vec3 sampleCosineHemisphere(const vec2& uv)
	{
		return sampleCosineHemisphere(uv.x, uv.y);
	}

	inline float rgbLuminance(const vec3& color)
	{
		return dot(vec3(0.2126f, 0.7152f, 0.0722f), color);
	}
}
