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
        HBasisT<vec3, L> hRadiance = {};
        const u32 sampleCount = (u32)data.m_radianceSamples.size();
        for (const RadianceSample& sample : data.m_radianceSamples)
        {
            hAddWeighted(hRadiance, hEvaluate<L>(sample.direction), sample.value  / (float)sampleCount);
        }

        HBasisT<vec3, L> hIrradiance = {};
        for (const RadianceSample& sample : data.m_irradianceSamples)
        {
            hAddWeighted(hIrradiance, hEvaluate<L>(sample.direction), sample.value / (float)sampleCount);
        }

        m_radianceImage = Image(data.m_outputSize);
        m_irradianceImage = Image(data.m_outputSize);

        data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
        {
            HBasisT<float, L> directionH = hEvaluate<L>(direction);

            vec3 sampleH = max(vec3(0.0f), hDot(hRadiance, directionH));
            m_radianceImage.at(pixelPos) = vec4(sampleH, 1.0f);

            vec3 sampleIrradianceH = max(vec3(0.0f), hDot(hRadiance, directionH));
            m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceH, 1.0f);
        });
    }
};

}