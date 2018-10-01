#include "SGBasis.h"
#include "Variance.h"

namespace Probulator
{
	vec3 sgBasisEvaluate(const SgBasis& basis, vec3 direction)
	{
		vec3 result = vec3(0.0f);
		for (const SphericalGaussian& basisLobe : basis)
		{
			result += sgEvaluate(basisLobe, direction);
		}
		return result;
	}

	vec3 sgBasisDot(const SgBasis& basis, const SphericalGaussian& lobe)
	{
		vec3 result = vec3(0.0f);
		for (const SphericalGaussian& basisLobe: basis)
		{
			result += sgDot(basisLobe, lobe);
		}
		return result;
	}

	float sgBasisNormalizationFactor(float lambda, u32 lobeCount)
	{
		// TODO: there is no solid basis for this right now
		float num = 4.0f * lambda * lambda;
		float den = (1.0f - exp(-2.0f*lambda)) * lobeCount;
		return num / den;
	}

	void sgBasisMeanAndVariance(const SphericalGaussian* lobes, u32 lobeCount, u32 sampleCount, vec3& outMean, vec3& outVariance)
	{
		OnlineVariance<vec3> accumulator;

		for (u32 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
		{
			vec2 sampleUv = sampleHammersley(sampleIt, sampleCount);
			vec3 direction = sampleUniformSphere(sampleUv);

			vec3 reconstructedValue = vec3(0.0f);
			for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
			{
				reconstructedValue += sgEvaluate(lobes[lobeIt], direction);
			}

			accumulator.addSample(reconstructedValue);
		}

		outMean = accumulator.mean;
		outVariance = accumulator.getVariance();
	}

	vec3 sgBasisMeanSquareError(const SgBasis& basis, const std::vector<RadianceSample>& radianceSamples)
	{
		vec3 errorSquaredSum = vec3(0.0f);

		for (const RadianceSample& sample : radianceSamples)
		{
			vec3 reconstructedValue = sgBasisEvaluate(basis, sample.direction);
			vec3 error = sample.value - reconstructedValue;
			errorSquaredSum += error*error;
		}

		float sampleWeight = 1.0f / radianceSamples.size();
		return errorSquaredSum * sampleWeight;
	}

	float sgBasisMeanSquareErrorScalar(const SgBasis& basis, const std::vector<RadianceSample>& radianceSamples)
	{
		return dot(sgBasisMeanSquareError(basis, radianceSamples), vec3(1.0f / 3.0f));
	}
    
    // Stephen Hill [2016], https://mynameismjp.wordpress.com/2016/10/09/sg-series-part-3-diffuse-lighting-from-an-sg-light-source/
    vec3 sgBasisIrradianceFitted(const SgBasis& basis, const vec3& normal)
    {
        vec3 result = vec3(0.0f);
        for (const SphericalGaussian& basisLobe : basis)
        {
            result += sgIrradianceFitted(basisLobe, normal);
        }
        return result / pi;
    }

}