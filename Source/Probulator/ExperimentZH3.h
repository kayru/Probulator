#pragma once

#include <Probulator/Experiments.h>
#include <ZH3Solver.h>

namespace Probulator
{

// ZH3 is an encoding that uses linear SH plus the quadratic SH zonal band in a direction determined by the linear SH.
// The extra component can either be explicitly encoded (for best quality) or hallucinated from the linear SH 
// (for improved irradiance reconstruction at no extra storage cost). 
// See https://research.activision.com/publications/2024/05/ZH3_QUADRATIC_ZONAL_HARMONICS for full details.

class ExperimentZH3: public Experiment
{
public:

	void run(SharedData& data) override
	{
		// Compute the input SH.
		SphericalHarmonicsL2RGB shRadiance = {};

		const ivec2 imageSize = data.m_outputSize;
		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			float texelArea = latLongTexelArea(pixelPos, imageSize);
			vec3 radiance = (vec3)data.m_radianceImage.at(pixelPos);
			shAddWeighted(shRadiance, shEvaluateL2(direction), radiance * texelArea);
		});

		m_radianceImage = Image(data.m_outputSize);
		m_irradianceImage = Image(data.m_outputSize);

		const float irradianceBandScales[9] = { 1.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f };

		Eigen::Matrix<float, 9, 3> eigenIrradiance;

		for (size_t i = 0; i < 9; i += 1)
		{
			eigenIrradiance.row(i) = Eigen::Vector3f(shRadiance[i].x, shRadiance[i].y, shRadiance[i].z) * irradianceBandScales[i];
		}

		// Use uniform weighting for the luminance so as to not bias the error towards any particular channel.
		const Eigen::Vector3f luminanceWeightingCoeffs = Eigen::Vector3f(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);

		// Solve to minimise the irradiance reconstruction error, not the radiance reconstruction error,
		// since this may affect the choice of axis.
		ZH3<float, 3> result;
		if (m_useSharedLuminanceAxis)
		{
			 result = ZH3SharedLuminanceSolver::solve(eigenIrradiance, luminanceWeightingCoeffs, 1.0f);
		}
		else
		{
			result = ZH3PerChannelSolver::solve(eigenIrradiance);
		}

		Eigen::Matrix<float, 9, 3> reconstructedSH3Irrad = result.expanded(luminanceWeightingCoeffs, m_useSharedLuminanceAxis ? 1.0f : 0.0f);

		SphericalHarmonicsL2RGB shIrradiance;

		for (size_t i = 0; i < 9; i += 1)
		{
			shIrradiance[i] = glm::vec3(reconstructedSH3Irrad(i, 0), reconstructedSH3Irrad(i, 1), reconstructedSH3Irrad(i, 2));
			shRadiance[i] = shIrradiance[i] * (1.0f / irradianceBandScales[i]);
		}

		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			SphericalHarmonicsL2 directionSh = shEvaluateL2(direction);

			vec3 sampleSh = max(vec3(0.0f), shDot(shRadiance, directionSh));
			m_radianceImage.at(pixelPos) = vec4(sampleSh, 1.0f);

			vec3 sampleIrradianceSh = max(vec3(0.0f), shDot(shIrradiance, directionSh));
			m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceSh, 1.0f);
		});
	}

	void getProperties(std::vector<Property>& outProperties) override
	{
		Experiment::getProperties(outProperties);
		outProperties.push_back(Property("Shared luminance axis", &m_useSharedLuminanceAxis));
	}

	ExperimentZH3& setUseSharedLuminanceAxis(bool useSharedAxis)
	{
		m_useSharedLuminanceAxis = useSharedAxis;
		return *this;
	}

	// Controls whether the solve uses a separate ZH3 axis per channel (0) or a single shared axis (1).
	// A single shared axis is cheaper to evaluate in shaders.
	bool m_useSharedLuminanceAxis = true;
};

class ExperimentHallucinateZH3: public Experiment
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

			vec3 sampleIrradianceSh = max(vec3(0.0f), shEvaluateDiffuseL1ZH3Hallucinate(shRadiance, direction) / pi);
			m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceSh, 1.0f);
		});
	}
};

// Like ExperimentHallucinateZH3, except we solve for the linear SH that will give the lowest reconstruction error with
// the hallucinated ZH3 coefficient.
class ExperimentSolveHallucinateZH3: public Experiment
{
public:
	void run(SharedData& data) override
	{
		SphericalHarmonicsL2RGB shRadiance = {};

		const ivec2 imageSize = data.m_outputSize;
		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			float texelArea = latLongTexelArea(pixelPos, imageSize);
			vec3 radiance = (vec3)data.m_radianceImage.at(pixelPos);
			shAddWeighted(shRadiance, shEvaluateL2(direction), radiance * texelArea);
		});

		m_radianceImage = Image(data.m_outputSize);
		m_irradianceImage = Image(data.m_outputSize);

		const float irradianceBandScales[9] = { 1.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f };

		Eigen::Matrix<float, 9, 3> eigenIrradiance;

		for (size_t i = 0; i < 9; i += 1)
		{
			eigenIrradiance.row(i) = Eigen::Vector3f(shRadiance[i].x, shRadiance[i].y, shRadiance[i].z) * irradianceBandScales[i];
		}

		// Use uniform weighting for the luminance so as to not bias the error towards any particular channel.
		const Eigen::Vector3f luminanceWeightingCoeffs = Eigen::Vector3f(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);

		// Solve to minimise the irradiance reconstruction error, not the radiance reconstruction error,
		// since this may affect the choice of axis.
		ZH3<float, 3> result;
		if (m_useSharedLuminanceAxis)
		{
			result = ZH3HallucinateSharedLuminanceSolver::solve(eigenIrradiance, /* targetIsIrradiance */ true, luminanceWeightingCoeffs, 1.0f);
		}
		else
		{
			result = ZH3HallucinatePerChannelSolver::solve(eigenIrradiance, /* targetIsIrradiance */ true);
		}

		// Note that the ZH3 coefficient in result can be recomputed from the linear coefficients;
		// see ZH3HallucinateSolver::hallucinateZH3.

		Eigen::Matrix<float, 9, 3> reconstructedSH3Irrad = result.expanded(luminanceWeightingCoeffs, m_useSharedLuminanceAxis ? 1.0f : 0.0f);

		SphericalHarmonicsL2RGB shIrradiance;

		for (size_t i = 0; i < 9; i += 1)
		{
			shIrradiance[i] = glm::vec3(reconstructedSH3Irrad(i, 0), reconstructedSH3Irrad(i, 1), reconstructedSH3Irrad(i, 2));
			shRadiance[i] = shIrradiance[i] * (1.0f / irradianceBandScales[i]);
		}

		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			SphericalHarmonicsL2 directionSh = shEvaluateL2(direction);

			vec3 sampleSh = max(vec3(0.0f), shDot(shRadiance, directionSh));
			m_radianceImage.at(pixelPos) = vec4(sampleSh, 1.0f);

			vec3 sampleIrradianceSh = max(vec3(0.0f), shDot(shIrradiance, directionSh));
			m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceSh, 1.0f);
		});
	}

	void getProperties(std::vector<Property>& outProperties) override
	{
		Experiment::getProperties(outProperties);
		outProperties.push_back(Property("Shared luminance axis", &m_useSharedLuminanceAxis));
	}

	ExperimentSolveHallucinateZH3& setUseSharedLuminanceAxis(bool useSharedAxis)
	{
		m_useSharedLuminanceAxis = useSharedAxis;
		return *this;
	}

	// Controls whether the solve uses a separate ZH3 axis per channel (0) or a single shared axis (1).
	// A single shared axis is cheaper to evaluate in shaders.
	bool m_useSharedLuminanceAxis = true;
};

}
