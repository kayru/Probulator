#pragma once

#include "Math.h"

namespace Probulator
{
	template <size_t L>
	struct SphericalHarmonicsT
	{
		static const size_t N = (L + 1)*(L + 1);
		float data[N];

		const float& operator [] (size_t i) const { return data[i]; }
		float& operator [] (size_t i) { return data[i]; }
	};

	typedef SphericalHarmonicsT<1> SphericalHarmonicsL1;
	typedef SphericalHarmonicsT<2> SphericalHarmonicsL2;

	struct SphericalHarmonicsL2RGB
	{
		SphericalHarmonicsL2 r, g, b;
	};

	SphericalHarmonicsL1 shEvaluateL1(vec3 p);
	SphericalHarmonicsL2 shEvaluateL2(vec3 p);
	float shEvaluateDiffuseL2(const SphericalHarmonicsL2& sh, vec3 n);
	void shReduceRingingL2(SphericalHarmonicsL2& sh, float lambda);
}
