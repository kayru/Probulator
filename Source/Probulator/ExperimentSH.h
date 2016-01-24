#pragma once

#include <Probulator/Experiments.h>

namespace Probulator
{
template <size_t L> class ExperimentSH : public Experiment
{
public:
	void run(SharedData& data) override
	{
		SphericalHarmonicsT<vec3, L> shRadiance = {};

		const ivec2 imageSize = data.m_outputSize;
		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			float texelArea = latLongTexelArea(pixelPos, imageSize);
			vec3 radiance = (vec3)data.m_radianceImage.at(pixelPos);
			shAddWeighted(shRadiance, shEvaluate<L>(direction), radiance * texelArea);
		});

		if (m_squaredLaplacianFraction < 1.0f)
		{
			SphericalHarmonicsT<float, L> sh = shLuminance(shRadiance);
			m_lambda = shFindWindowingLambda(sh, m_squaredLaplacianFraction);
		}

		if (m_lambda != 0.0f)
		{
			shApplyWindowing<vec3, L>(shRadiance, m_lambda);
		}

		m_radianceImage = Image(data.m_outputSize);
		m_irradianceImage = Image(data.m_outputSize);

		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			SphericalHarmonicsT<float, L> shDirection = shEvaluate<L>(direction);

			vec3 sampleSh = max(vec3(0.0f), shDot(shRadiance, shDirection));
			m_radianceImage.at(pixelPos) = vec4(sampleSh, 1.0f);

			vec3 sampleIrradianceSh = max(vec3(0.0f), shEvaluateDiffuse<vec3, L>(shRadiance, direction) / pi);
			m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceSh, 1.0f);
		});
	}

	void getProperties(std::vector<Property>& outProperties) override
	{
		Experiment::getProperties(outProperties);
		outProperties.push_back(Property("Lambda", &m_lambda));
		outProperties.push_back(Property("Squared Laplacian fraction", &m_squaredLaplacianFraction));
	}

	ExperimentSH<L>& setLambda(float v)
	{
		m_lambda = v;
		return *this;
	}

	ExperimentSH<L>& setSquaredLaplacianFraction(float v)
	{
		m_squaredLaplacianFraction = v;
		return *this;
	}

	float m_lambda = 0.0f;
	float m_squaredLaplacianFraction = 1.0f; // Expressed as a fraction of the original squared Laplacian
};

class ExperimentSHL1Geomerics : public Experiment
{
public:
	void run(SharedData& data) override
	{
		SphericalHarmonicsL1RGB shRadiance = {};

		const ivec2 imageSize = data.m_outputSize;
		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			float texelArea = latLongTexelArea(pixelPos, imageSize);
			vec3 radiance = (vec3)data.m_radianceImage.at(pixelPos);
			shAddWeighted(shRadiance, shEvaluateL1(direction), radiance * texelArea);
		});

		m_radianceImage = Image(data.m_outputSize);
		m_irradianceImage = Image(data.m_outputSize);

		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			SphericalHarmonicsL1 directionSh = shEvaluateL1(direction);

			vec3 sampleSh = max(vec3(0.0f), shDot(shRadiance, directionSh));
			m_radianceImage.at(pixelPos) = vec4(sampleSh, 1.0f);

			vec3 sampleIrradianceSh;
			for (u32 i = 0; i < 3; ++i)
			{
				SphericalHarmonicsL1 shRadianceChannel;
				shRadianceChannel[0] = shRadiance[0][i];
				shRadianceChannel[1] = shRadiance[1][i];
				shRadianceChannel[2] = shRadiance[2][i];
				shRadianceChannel[3] = shRadiance[3][i];
				sampleIrradianceSh[i] = shEvaluateDiffuseL1Geomerics(shRadianceChannel, direction) / pi;
			}
			m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceSh, 1.0f);
		});
	}
};
}
