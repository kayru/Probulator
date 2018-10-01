#pragma once

#include "SphericalGaussian.h"
#include "RadianceSample.h"

#include <vector>

namespace Probulator
{
	typedef std::vector<SphericalGaussian> SgBasis;

	float sgBasisNormalizationFactor(float lambda, u32 lobeCount);
	vec3 sgBasisEvaluate(const SgBasis& basis, vec3 direction);
	vec3 sgBasisDot(const SgBasis& basis, const SphericalGaussian& lobe);
	void sgBasisMeanAndVariance(const SphericalGaussian* lobes, u32 lobeCount, u32 sampleCount, vec3& outMean, vec3& outVariance);
	vec3 sgBasisMeanSquareError(const SgBasis& basis, const std::vector<RadianceSample>& radianceSamples);
	float sgBasisMeanSquareErrorScalar(const SgBasis& basis, const std::vector<RadianceSample>& radianceSamples);
    vec3 sgBasisIrradianceFitted(const SgBasis& basis, const vec3& normal);
}
