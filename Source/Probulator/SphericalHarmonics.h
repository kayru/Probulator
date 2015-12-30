#pragma once

#include "Math.h"

namespace Probulator
{
	// https://graphics.stanford.edu/papers/envmap/envmap.pdf

	template <typename T, size_t L>
	struct SphericalHarmonicsT
	{
		T data[(L + 1)*(L + 1)];

		const T& operator [] (size_t i) const { return data[i]; }
		T& operator [] (size_t i) { return data[i]; }
	};

	typedef SphericalHarmonicsT<float, 1> SphericalHarmonicsL1;
	typedef SphericalHarmonicsT<float, 2> SphericalHarmonicsL2;
	typedef SphericalHarmonicsT<vec3, 1> SphericalHarmonicsL1RGB;
	typedef SphericalHarmonicsT<vec3, 2> SphericalHarmonicsL2RGB;

	SphericalHarmonicsL1 shEvaluateL1(vec3 p);
	SphericalHarmonicsL2 shEvaluateL2(vec3 p);
	float shEvaluateDiffuseL2(const SphericalHarmonicsL2& sh, vec3 n);
	void shReduceRingingL2(SphericalHarmonicsL2& sh, float lambda);

	inline size_t shSize(size_t L) { return (L + 1)*(L + 1); }

	template <typename Ta, typename Tb, typename Tw, size_t L>
	inline void shAddWeighted(SphericalHarmonicsT<Ta, L>& accumulatorSh, const SphericalHarmonicsT<Tb, L>& sh, const Tw& weight)
	{
		for (size_t i = 0; i < shSize(L); ++i)
		{
			accumulatorSh[i] += sh[i] * weight;
		}
	}

	template <typename Ta, typename Tb, size_t L>
	inline Ta shDot(const SphericalHarmonicsT<Ta, L>& shA, const SphericalHarmonicsT<Tb, L>& shB)
	{
		Ta result = Ta(0);
		for (size_t i = 0; i < shSize(L); ++i)
		{
			result += shA[i] * shB[i];
		}
		return result;
	}

	inline SphericalHarmonicsL1 shEvaluateL1(vec3 p)
	{
		float c1 = 0.282095f;
		float c2 = 0.488603f;

		SphericalHarmonicsL1 sh;

		sh[0] = c1; // Y00

		sh[1] = c2 * p.y; // Y1-1
		sh[2] = c2 * p.z; // Y10
		sh[3] = c2 * p.x; // Y1+1

		return sh;
	}

	inline SphericalHarmonicsL2 shEvaluateL2(vec3 p)
	{
		float c1 = 0.282095f;
		float c2 = 0.488603f;
		float c3 = 1.092548f;
		float c4 = 0.315392f;
		float c5 = 0.546274f;

		SphericalHarmonicsL2 sh;

		sh[0] = c1; // Y00

		sh[1] = c2 * p.y; // Y1-1
		sh[2] = c2 * p.z; // Y10
		sh[3] = c2 * p.x; // Y1+1

		sh[4] = c3 * p.x * p.y; // Y2-2
		sh[5] = c3 * p.y * p.z; // Y2-1
		sh[6] = c4 * (3.0f * p.z * p.z - 1.0f); // Y20
		sh[7] = c3 * p.x * p.z; // Y2+1
		sh[8] = c5 * (p.x * p.x - p.y * p.y); // Y2+2

		return sh;
	}

	template <typename T>
	inline T shEvaluateDiffuseL1(const SphericalHarmonicsT<T, 1>& sh, const vec3& direction)
	{
		float c1 = 0.886227f;
		float c2 = 2.0f * 0.511664f;

		const T& L00 = sh[0];

		const T& L1_1 = sh[1];
		const T& L10 = sh[2];
		const T& L11 = sh[3];

		float x = direction.x;
		float y = direction.y;
		float z = direction.z;

		return c1*L00 + c2 * (L11*x + L1_1*y + L10*z);
	}

	template <typename T>
	inline T shEvaluateDiffuseL2(const SphericalHarmonicsT<T, 2>& sh, const vec3& direction)
	{
		float c1 = 0.429043f;
		float c2 = 0.511664f;
		float c3 = 0.743125f;
		float c4 = 0.886227f;
		float c5 = 0.247708f;

		const T& L00 = sh[0];

		const T& L1_1 = sh[1];
		const T& L10 = sh[2];
		const T& L11 = sh[3];

		const T& L2_2 = sh[4];
		const T& L2_1 = sh[5];
		const T& L20 = sh[6];
		const T& L21 = sh[7];
		const T& L22 = sh[8];

		float x = direction.x;
		float y = direction.y;
		float z = direction.z;

		float x2 = x * x;
		float y2 = y * y;
		float z2 = z * z;

		return c1*L22*(x2 - y2)
			+ c3*L20*z2
			+ c4*L00
			- c5*L20
			+ 2.0f * c1 * (L2_2*x*y + L21*x*z + L2_1*y*z)
			+ 2.0f * c2 * (L11*x + L1_1*y + L10*z);
	}

	template <typename T, size_t L>
	void shReduceRinging(SphericalHarmonicsT<T, L>& sh, float lambda)
	{
		// From Peter-Pike Sloan's Stupid SH Tricks
		// http://www.ppsloan.org/publications/StupidSH36.pdf

		auto scale = [lambda](u32 l) { return 1.0f / (1.0f + lambda * l * l * (l + 1.0f) * (l + 1.0f)); };

		sh[0] *= scale(0);

		sh[1] *= scale(1);
		sh[2] *= scale(1);
		sh[3] *= scale(1);

		sh[4] *= scale(2);
		sh[5] *= scale(2);
		sh[6] *= scale(2);
		sh[7] *= scale(2);
		sh[8] *= scale(2);
	}

	inline vec3 shMeanSquareError(const SphericalHarmonicsL2RGB& sh, const std::vector<RadianceSample>& radianceSamples)
	{
		vec3 errorSquaredSum = vec3(0.0f);

		for (const RadianceSample& sample : radianceSamples)
		{
			SphericalHarmonicsL2 directionSh = shEvaluateL2(sample.direction);
			vec3 reconstructedValue = shDot(sh, directionSh);
			vec3 error = sample.value - reconstructedValue;
			errorSquaredSum += error*error;
		}

		float sampleWeight = 1.0f / radianceSamples.size();
		return errorSquaredSum * sampleWeight;
	}

	inline float shMeanSquareErrorScalar(const SphericalHarmonicsL2RGB& sh, const std::vector<RadianceSample>& radianceSamples)
	{
		return dot(shMeanSquareError(sh, radianceSamples), vec3(1.0f / 3.0f));
	}

	inline vec3 shMeanSquareError(const SphericalHarmonicsL1RGB& sh, const std::vector<RadianceSample>& radianceSamples)
	{
		vec3 errorSquaredSum = vec3(0.0f);

		for (const RadianceSample& sample : radianceSamples)
		{
			SphericalHarmonicsL1 directionSh = shEvaluateL1(sample.direction);
			vec3 reconstructedValue = shDot(sh, directionSh);
			vec3 error = sample.value - reconstructedValue;
			errorSquaredSum += error*error;
		}

		float sampleWeight = 1.0f / radianceSamples.size();
		return errorSquaredSum * sampleWeight;
	}

	inline float shMeanSquareErrorScalar(const SphericalHarmonicsL1RGB& sh, const std::vector<RadianceSample>& radianceSamples)
	{
		return dot(shMeanSquareError(sh, radianceSamples), vec3(1.0f / 3.0f));
	}
}
