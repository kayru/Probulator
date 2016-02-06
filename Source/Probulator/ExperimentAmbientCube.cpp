#include "ExperimentAmbientCube.h"

#include <Eigen/Eigen>
#include <Eigen/nnls.h>

namespace Probulator {

ExperimentAmbientCube::AmbientCube ExperimentAmbientCube::solveAmbientCube(const ImageBase<vec3>& directions, const Image& irradiance)
{
	using namespace Eigen;

	AmbientCube ambientCube;

	const u64 sampleCount = directions.getPixelCount();

	MatrixXf A;
	A.resize(sampleCount, 6);

	for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
	{
		const vec3& direction = directions.at(sampleIt);
		vec3 dirSquared = direction * direction;

		if (direction.x < 0)
		{
			A(sampleIt, 0) = dirSquared.x;
			A(sampleIt, 1) = 0.0f;
		}
		else
		{
			A(sampleIt, 0) = 0.0f;
			A(sampleIt, 1) = dirSquared.x;
		}

		if (direction.y < 0)
		{
			A(sampleIt, 2) = dirSquared.y;
			A(sampleIt, 3) = 0.0f;
		}
		else
		{
			A(sampleIt, 2) = 0.0f;
			A(sampleIt, 3) = dirSquared.y;
		}

		if (direction.z < 0)
		{
			A(sampleIt, 4) = dirSquared.z;
			A(sampleIt, 5) = 0.0f;
		}
		else
		{
			A(sampleIt, 4) = 0.0f;
			A(sampleIt, 5) = dirSquared.z;
		}
	}

	NNLS<MatrixXf> solver(A);

	VectorXf b;
	b.resize(sampleCount);

	for (u32 channelIt = 0; channelIt < 3; ++channelIt)
	{
		for (u64 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
		{
			b[sampleIt] = irradiance.at(sampleIt)[channelIt];
		}

		solver.solve(b);
		VectorXf x = solver.x();

		for (u64 basisIt = 0; basisIt < 6; ++basisIt)
		{
			ambientCube.irradiance[basisIt][channelIt] = x[basisIt];
		}
	}

	return ambientCube;
}

void ExperimentAmbientCube::run(SharedData& data)
{
	AmbientCube ambientCube = solveAmbientCube(data.m_directionImage, m_input->m_irradianceImage);

	m_radianceImage = Image(data.m_outputSize);
	m_irradianceImage = Image(data.m_outputSize);

	data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
	{
		vec3 sampleIrradianceH = ambientCube.evaluate(direction);
		m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceH, 1.0f);
		m_radianceImage.at(pixelPos) = vec4(0.0f);
	});
}

}
