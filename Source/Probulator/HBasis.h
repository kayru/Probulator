#pragma once

#include "Math.h"

namespace Probulator
{
	// https://www.cg.tuwien.ac.at/research/publications/2010/Habel-2010-EIN/Habel-2010-EIN-paper.pdf

	template <typename T, size_t L>
	struct HBasisT
	{
		T data[L];

		const T& operator [] (size_t i) const { return data[i]; }
		T& operator [] (size_t i) { return data[i]; }
	};

	typedef HBasisT<float, 4> HBasis4;
	typedef HBasisT<float, 6> HBasis6;
	
	template <typename T, size_t L> 
	HBasis4 hEvaluate4(vec3 p);
	HBasis6 hEvaluate6(vec3 p);
	
	inline size_t hSize(size_t L) { return L; }

	template <typename Ta, typename Tb, typename Tw, size_t L>
	inline void hAddWeighted(HBasisT<Ta, L>& accumulator, const HBasisT<Tb, L>& h, const Tw& weight)
	{
		for (size_t i = 0; i < hSize(L); ++i)
		{
			accumulator[i] += h[i] * weight;
		}
	}

	template <typename Ta, typename Tb, size_t L>
	inline Ta hDot(const HBasisT<Ta, L>& hA, const HBasisT<Tb, L>& hB)
	{
		Ta result = Ta(0);
		for (size_t i = 0; i < hSize(L); ++i)
		{
			result += hA[i] * hB[i];
		}
		return result;
	}

	template <size_t L>
	inline HBasisT<float, L> hEvaluate(vec3 p)
	{
		HBasisT<float, L> result;

        if(p.z < 0.0f) {
            for(size_t i = 0; i < L; ++i)
                result[i] = 0.0f;

            return result;
        }

		const float x = -p.x;
		const float y = -p.y;
		const float z = p.z;

		const float x2 = x*x;
		const float y2 = y*y;
		const float z2 = z*z;

		const float z3 = z2*z;

		const float x4 = x2*x2;
		const float y4 = y2*y2;
		const float z4 = z2*z2;

		const float sqrtPi = sqrt(pi);

		size_t i = 0;

		result[i++] =  1.0f/(2.0f*sqrtPi);

		if (L >= 4)
		{
			result[i++] = -sqrt(3.0f/(2.0f*pi))*y;
			result[i++] =  sqrt(3.0f/(2.0f*pi))*(2*z - 1.0f);
			result[i++] = -sqrt(3.0f/(2.0f*pi))*x;
		}

		if (L >= 6)
		{
			result[i++] =        sqrt(15.0f / (2.0f*pi))*x*y;
			result[i++] = 0.5f * sqrt(15.0f / (2.0f*pi))*(x2 - y2);
		}

		return result;
	}

	inline HBasis4 hEvaluate4(vec3 p)
	{
		return hEvaluate<4>(p);
	}

	inline HBasis6 hEvaluate6(vec3 p)
	{
		return hEvaluate<6>(p);
	}
	
	template <typename T, size_t L>
	inline T hMeanSquareError(const HBasisT<T, L>& h, const std::vector<RadianceSample>& radianceSamples)
	{
		T errorSquaredSum = T(0.0f);

		for (const RadianceSample& sample : radianceSamples)
		{
			auto directionH = hEvaluate<L>(sample.direction);
			auto reconstructedValue = hDot(h, directionH);
			auto error = sample.value - reconstructedValue;
			errorSquaredSum += error*error;
		}

		float sampleWeight = 1.0f / radianceSamples.size();
		return errorSquaredSum * sampleWeight;
	}

	template <typename T, size_t L>
	inline float hMeanSquareErrorScalar(const HBasisT<T, L>& sh, const std::vector<RadianceSample>& radianceSamples)
	{
		return dot(hMeanSquareError(sh, radianceSamples), T(1.0f / 3.0f));
	}
}