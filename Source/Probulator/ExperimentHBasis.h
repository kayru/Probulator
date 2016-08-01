#pragma once

#include <Probulator/Experiments.h>

namespace Probulator {

template <size_t L>
class ExperimentHBasis : public Experiment
{
public:

	ExperimentHBasis()
	{
		m_isHemispherical = true;
	}

	void run(SharedData& data) override
	{
		SphericalHarmonicsT<vec3, 2> shRadiance = {};

		const ivec2 imageSize = data.m_outputSize;
		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			float texelArea = latLongTexelArea(pixelPos, imageSize);
			vec3 radiance = (vec3)data.m_radianceImage.at(pixelPos);
			shAddWeighted(shRadiance, shEvaluate<2>(direction), radiance * texelArea);
		});

		SphericalHarmonicsT<vec3, 2> shIrradiance = shConvolveDiffuse(shRadiance);

		HBasisT<vec3, L> hRadiance = convertShToHBasis<L>(shRadiance);
		HBasisT<vec3, L> hIrradiance = convertShToHBasis<L>(shIrradiance);

		m_radianceImage = Image(data.m_outputSize);
		m_irradianceImage = Image(data.m_outputSize);

		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			HBasisT<float, L> directionH = hEvaluate<L>(direction);

			vec3 sampleH = max(vec3(0.0f), hDot(hRadiance, directionH));
			m_radianceImage.at(pixelPos) = vec4(sampleH, 1.0f);

			vec3 sampleIrradianceH = max(vec3(0.0f), hDot(hIrradiance, directionH)) / pi;
			m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceH, 1.0f);
		});
	}
};

}
