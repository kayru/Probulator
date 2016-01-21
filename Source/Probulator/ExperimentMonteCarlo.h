#pragma once

#include <Probulator/Experiments.h>

namespace Probulator {

class ExperimentMC : public Experiment
{
public:

    void run(SharedData& data) override
    {
        const ivec2 imageSize = data.m_outputSize;
        const ivec2 imageSizeMinusOne = imageSize - 1;
        const vec2 imageSizeMinusOneRcp = vec2(1.0) / vec2(imageSizeMinusOne);

        m_radianceImage = data.m_radianceImage;

        m_irradianceImage = Image(data.m_outputSize);
        m_irradianceImage.parallelForPixels2D([&](vec4& pixel, ivec2 pixelPos)
        {
            vec2 uv = (vec2(pixelPos) + vec2(0.5f)) * imageSizeMinusOneRcp;
            vec3 direction = latLongTexcoordToCartesian(uv);
            mat3 basis = makeOrthogonalBasis(direction);
            vec3 accum = vec3(0.0f);
            for (u32 sampleIt = 0; sampleIt < m_hemisphereSampleCount; ++sampleIt)
            {
                vec2 sampleUv = sampleHammersley(sampleIt, m_hemisphereSampleCount);
                vec3 hemisphereDirection = sampleCosineHemisphere(sampleUv);
                vec3 sampleDirection = basis * hemisphereDirection;
                accum += (vec3)m_radianceImage.sampleNearest(cartesianToLatLongTexcoord(sampleDirection));
            }

            accum /= m_hemisphereSampleCount;

            pixel = vec4(accum, 1.0f);
        });
    }

	void getProperties(std::vector<Property>& outProperties) override
	{
		Experiment::getProperties(outProperties);
		outProperties.push_back(Property("Hemisphere sample count", reinterpret_cast<int*>(&m_hemisphereSampleCount)));
	}

    ExperimentMC& setHemisphereSampleCount(u32 v) { m_hemisphereSampleCount = v; return *this; }

    u32 m_hemisphereSampleCount = 1000;
};

class ExperimentMCIS : public Experiment
{
public:

    void run(SharedData& data) override
    {
        const ivec2 imageSize = data.m_outputSize;
        const ivec2 imageSizeMinusOne = imageSize - 1;
        const vec2 imageSizeMinusOneRcp = vec2(1.0) / vec2(imageSizeMinusOne);

        m_radianceImage = data.m_radianceImage;

        std::vector<float> texelWeights;
        std::vector<float> texelAreas;
        float weightSum = 0.0;
        m_radianceImage.forPixels2D([&](const vec4& p, ivec2 pixelPos)
        {
            float area = latLongTexelArea(pixelPos, imageSize);

            float intensity = dot(vec3(1.0f / 3.0f), (vec3)p);
            float weight = intensity * area;

            weightSum += weight;
            texelWeights.push_back(weight);
            texelAreas.push_back(area);
        });

        DiscreteDistribution<float> discreteDistribution(texelWeights.data(), texelWeights.size(), weightSum);

        m_irradianceImage = Image(data.m_outputSize);
        m_irradianceImage.parallelForPixels2D([&](vec4& pixel, ivec2 pixelPos)
        {
            u32 pixelIndex = pixelPos.x + pixelPos.y * m_irradianceImage.getWidth();
            u32 seed = m_jitterEnabled ? pixelIndex : 0;
            std::mt19937 rng(seed);

            vec2 uv = (vec2(pixelPos) + vec2(0.5f)) * imageSizeMinusOneRcp;
            vec3 normal = latLongTexcoordToCartesian(uv);
            vec3 accum = vec3(0.0f);
            for (u32 sampleIt = 0; sampleIt < m_sampleCount; ++sampleIt)
            {
                u32 sampleIndex = (u32)discreteDistribution(rng);
                float sampleProbability = texelWeights[sampleIndex] / weightSum;
                vec3 sampleDirection = data.m_directionImage.at(sampleIndex);
                float cosTerm = dotMax0(normal, sampleDirection);
                float sampleArea = (float)texelAreas[sampleIndex];
                vec3 sampleRadiance = (vec3)m_radianceImage.at(sampleIndex) * sampleArea;
                accum += sampleRadiance * cosTerm / sampleProbability;
            }

            accum /= m_sampleCount * pi;

            pixel = vec4(accum, 1.0f);
        });

        data.GenerateIrradianceSamples(m_irradianceImage);
    }

	void getProperties(std::vector<Property>& outProperties) override
	{
		Experiment::getProperties(outProperties);
		outProperties.push_back(Property("Sample count", reinterpret_cast<int*>(&m_sampleCount)));
		outProperties.push_back(Property("Jitter enabled", &m_jitterEnabled));
	}

    ExperimentMCIS& setSampleCount(u32 v)
    {
        m_sampleCount = v;
        return *this;
    }

    ExperimentMCIS& setJitterEnabled(bool state)
    {
        m_jitterEnabled = state;
        return *this;
    }

    u32 m_sampleCount = 1000;
    bool m_jitterEnabled = false;
};

}