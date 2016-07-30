#include "ExperimentRNM.h"

namespace Probulator {

void ExperimentRNM::run(SharedData& data)
{
	RNMBasis rnm;

	for(u32 i=0; i<3; ++i)
	{
		vec2 texcoord = cartesianToLatLongTexcoord(RNMBasis::getVector(i));
		rnm.irradiance[i] = (vec3)m_input->m_irradianceImage.sampleNearest(texcoord);
	}

	m_radianceImage = Image(data.m_outputSize);
	m_irradianceImage = Image(data.m_outputSize);

	data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
	{
		vec3 sampleIrradianceH = rnm.evaluate(direction);
		m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceH, 1.0f);
		m_radianceImage.at(pixelPos) = vec4(0.0f);
	});
}

}