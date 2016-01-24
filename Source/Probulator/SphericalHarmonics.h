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

		T& at(int l, int m) { return data[l * l + l + m]; }
		const T& at(int l, int m) const { return data[l * l + l + m]; }
	};

	typedef SphericalHarmonicsT<float, 1> SphericalHarmonicsL1;
	typedef SphericalHarmonicsT<float, 2> SphericalHarmonicsL2;
	typedef SphericalHarmonicsT<vec3, 1> SphericalHarmonicsL1RGB;
	typedef SphericalHarmonicsT<vec3, 2> SphericalHarmonicsL2RGB;

	template <typename T, size_t L> 
	SphericalHarmonicsL1 shEvaluateL1(vec3 p);
	SphericalHarmonicsL2 shEvaluateL2(vec3 p);

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

	template <size_t L>
	inline SphericalHarmonicsT<float, L> shEvaluate(vec3 p)
	{
		// From Peter-Pike Sloan's Stupid SH Tricks
		// http://www.ppsloan.org/publications/StupidSH36.pdf
		// https://github.com/dariomanesku/cmft/blob/master/src/cmft/cubemapfilter.cpp#L130

		static_assert(L<=4, "Spherical Harmonics above L4 are not supported");

		SphericalHarmonicsT<float, L> result;

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

		if (L >= 1)
		{
			result[i++] = -sqrt(3.0f/(4.0f*pi))*y;
			result[i++] =  sqrt(3.0f/(4.0f*pi))*z;
			result[i++] = -sqrt(3.0f/(4.0f*pi))*x;        	
		}

		if (L >= 2)
		{
			result[i++] =  sqrt(15.0f/(4.0f*pi))*y*x;
			result[i++] = -sqrt(15.0f/(4.0f*pi))*y*z;
			result[i++] =  sqrt(5.0f/(16.0f*pi))*(3.0f*z2-1.0f);
			result[i++] = -sqrt(15.0f/(4.0f*pi))*x*z;
			result[i++] =  sqrt(15.0f/(16.0f*pi))*(x2-y2);			
		}

		if (L >= 3)
		{
			result[i++] = -sqrt( 70.0f/(64.0f*pi))*y*(3.0f*x2-y2);
			result[i++] =  sqrt(105.0f/ (4.0f*pi))*y*x*z;
			result[i++] = -sqrt( 21.0f/(16.0f*pi))*y*(-1.0f+5.0f*z2);
			result[i++] =  sqrt(  7.0f/(16.0f*pi))*(5.0f*z3-3.0f*z);
			result[i++] = -sqrt( 42.0f/(64.0f*pi))*x*(-1.0f+5.0f*z2);
			result[i++] =  sqrt(105.0f/(16.0f*pi))*(x2-y2)*z;
			result[i++] = -sqrt( 70.0f/(64.0f*pi))*x*(x2-3.0f*y2);			
		}

		if (L >= 4)
		{
			result[i++] =  3.0f*sqrt(35.0f/(16.0f*pi))*x*y*(x2-y2);
			result[i++] = -3.0f*sqrt(70.0f/(64.0f*pi))*y*z*(3.0f*x2-y2);
			result[i++] =  3.0f*sqrt( 5.0f/(16.0f*pi))*y*x*(-1.0f+7.0f*z2);
			result[i++] = -3.0f*sqrt(10.0f/(64.0f*pi))*y*z*(-3.0f+7.0f*z2);
			result[i++] =  (105.0f*z4-90.0f*z2+9.0f)/(16.0f*sqrtPi);
			result[i++] = -3.0f*sqrt(10.0f/(64.0f*pi))*x*z*(-3.0f+7.0f*z2);
			result[i++] =  3.0f*sqrt( 5.0f/(64.0f*pi))*(x2-y2)*(-1.0f+7.0f*z2);
			result[i++] = -3.0f*sqrt(70.0f/(64.0f*pi))*x*z*(x2-3.0f*y2);
			result[i++] =  3.0f*sqrt(35.0f/(4.0f*(64.0f*pi)))*(x4-6.0f*y2*x2+y4);			
		}

		return result;
	}

	inline SphericalHarmonicsL1 shEvaluateL1(vec3 p)
	{
		return shEvaluate<1>(p);
	}

	inline SphericalHarmonicsL2 shEvaluateL2(vec3 p)
	{
		return shEvaluate<2>(p);
	}

	inline float shEvaluateDiffuseL1Geomerics(const SphericalHarmonicsL1& sh, const vec3& n)
	{
		// http://www.geomerics.com/wp-content/uploads/2015/08/CEDEC_Geomerics_ReconstructingDiffuseLighting1.pdf

		float R0 = sh[0];

		vec3 R1 = 0.5f * vec3(sh[3], sh[1], sh[2]);
		float lenR1 = length(R1);

		float q = 0.5f * (1.0f + dot(R1 / lenR1, n));

		float p = 1.0f + 2.0f * lenR1 / R0;
		float a = (1.0f - lenR1 / R0) / (1.0f + lenR1 / R0);

		return R0 * (a + (1.0f - a) * (p + 1.0f) * pow(q, p));
	}

	template <typename T, size_t L>
	inline SphericalHarmonicsT<T, L> shConvolveDiffuse(SphericalHarmonicsT<T, L>& sh)
	{
		SphericalHarmonicsT<T, L> result;

		// https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf equation 8

		const float A[5] = {
			pi,
			pi * 2.0f / 3.0f,
			pi * 1.0f / 4.0f,
			0.0f,
			-pi * 1.0f / 24.0f
		};

		int i = 0;
		for (int l = 0; l <= (int)L; ++l)
		{
			for (int m = -l; m <= l; ++m)
			{
				result[i] = sh[i] * A[l];
				++i;
			}
		}

		return result;
	}

	template <typename T, size_t L>
	inline T shEvaluateDiffuse(const SphericalHarmonicsT<T, L>& sh, const vec3& direction)
	{
		static_assert(L<=4, "Spherical Harmonics above L4 are not supported");

		SphericalHarmonicsT<float, L> directionSh = shEvaluate<L>(direction);

		// https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf equation 8

		const float A[5] = {
			pi,
			pi * 2.0f / 3.0f,
			pi * 1.0f / 4.0f,
			0.0f,
			-pi * 1.0f / 24.0f
		};

		size_t i = 0;

		T result = sh[i] * directionSh[i] * A[0]; ++i;

		if (L >= 1)
		{
			result += sh[i] * directionSh[i] * A[1]; ++i;
			result += sh[i] * directionSh[i] * A[1]; ++i;
			result += sh[i] * directionSh[i] * A[1]; ++i;
		}

		if (L >= 2)
		{
			result += sh[i] * directionSh[i] * A[2]; ++i;
			result += sh[i] * directionSh[i] * A[2]; ++i;
			result += sh[i] * directionSh[i] * A[2]; ++i;
			result += sh[i] * directionSh[i] * A[2]; ++i;
			result += sh[i] * directionSh[i] * A[2]; ++i;
		}

		// L3 and other odd bands > 1 have 0 factor

		if (L >= 4)
		{
			i = 16;

			result += sh[i] * directionSh[i] * A[4]; ++i;
			result += sh[i] * directionSh[i] * A[4]; ++i;
			result += sh[i] * directionSh[i] * A[4]; ++i;
			result += sh[i] * directionSh[i] * A[4]; ++i;
			result += sh[i] * directionSh[i] * A[4]; ++i;
			result += sh[i] * directionSh[i] * A[4]; ++i;
			result += sh[i] * directionSh[i] * A[4]; ++i;
			result += sh[i] * directionSh[i] * A[4]; ++i;
			result += sh[i] * directionSh[i] * A[4]; ++i;
		}

		return result;
	}


	template <typename T>
	inline T shEvaluateDiffuseL1(const SphericalHarmonicsT<T, 1>& sh, const vec3& direction)
	{
		return shEvaluateDiffuse<T, 1>(sh, direction);
	}

	template <typename T>
	inline T shEvaluateDiffuseL2(const SphericalHarmonicsT<T, 2>& sh, const vec3& direction)
	{
		return shEvaluateDiffuse<T, 2>(sh, direction);
	}

	template <size_t L>
	float shFindWindowingLambda(const SphericalHarmonicsT<float, L>& sh, float squaredLaplacianFraction)
	{
		// http://www.ppsloan.org/publications/StupidSH36.pdf
		// Appendix A7 Solving for Lambda to Reduce the Squared Laplacian

		float tableL[L + 1];
		float tableB[L + 1];

		tableL[0] = 0.0f;
		tableB[0] = 0.0f;

		for (int l = 1; l <= (int)L; ++l)
		{
			tableL[l] = float(sqr(l) * sqr(l + 1));

			float B = 0.0f;
			for (int m = -1; m <= l; ++m)
			{
				B += sqr(sh.at(l, m));
			}
			tableB[l] = B;
		}

		float squaredLaplacian = 0.0f;

		for (int l = 1; l <= (int)L; ++l)
		{
			squaredLaplacian += tableL[l] * tableB[l];
		}

		const float targetSquaredLaplacian = squaredLaplacian * squaredLaplacianFraction;

		float lambda = 0.0f;

		const u32 iterationLimit = 10000000;
		for (u32 i = 0; i < iterationLimit; ++i)
		{
			float f = 0.0f;
			float fd = 0.0f;

			for (int l = 1; l <= (int)L; ++l)
			{
				f += tableL[l] * tableB[l] / sqr(1.0f + lambda * tableL[l]);
				fd += (2.0f * sqr(tableL[l]) * tableB[l]) / cube(1.0f + lambda * tableL[l]);
			}

			f = targetSquaredLaplacian - f;

			float delta = -f / fd;
			lambda += delta;

			if (abs(delta) < 1e-6f)
			{
				break;
			}
		}

		return lambda;
	}

	template <typename T, size_t L>
	void shApplyWindowing(SphericalHarmonicsT<T, L>& sh, float lambda)
	{
		// From Peter-Pike Sloan's Stupid SH Tricks
		// http://www.ppsloan.org/publications/StupidSH36.pdf

		int i = 0;
		for(int l = 0; l <= (int)L; ++l)
		{
			float s = 1.0f / (1.0f + lambda * l * l * (l + 1.0f) * (l + 1.0f));
			for(int m = -l; m <= l; ++m)
			{
				sh[i++] *= s;
			}
		}
	}

	template <typename T, size_t L>
	inline T shMeanSquareError(const SphericalHarmonicsT<T, L>& sh, const std::vector<RadianceSample>& radianceSamples)
	{
		T errorSquaredSum = T(0.0f);

		for (const RadianceSample& sample : radianceSamples)
		{
			auto directionSh = shEvaluate<L>(sample.direction);
			auto reconstructedValue = shDot(sh, directionSh);
			auto error = sample.value - reconstructedValue;
			errorSquaredSum += error*error;
		}

		float sampleWeight = 1.0f / radianceSamples.size();
		return errorSquaredSum * sampleWeight;
	}

	template <typename T, size_t L>
	inline float shMeanSquareErrorScalar(const SphericalHarmonicsT<T, L>& sh, const std::vector<RadianceSample>& radianceSamples)
	{
		return dot(shMeanSquareError(sh, radianceSamples), T(1.0f / 3.0f));
	}

	template <size_t L>
	inline SphericalHarmonicsT<float, L> shLuminance(const SphericalHarmonicsT<vec3, L>& sh)
	{
		SphericalHarmonicsT<float, L> result;
		for (size_t i = 0; i < shSize(L); ++i)
		{
			result[i] = rgbLuminance(sh[i]);
		}
		return result;
	}
}
