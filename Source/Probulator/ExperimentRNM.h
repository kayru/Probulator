#pragma once

#include <Probulator/Experiments.h>

namespace Probulator {

// Radiosity Normal Mapping
// http://www.valvesoftware.com/publications/2006/SIGGRAPH06_Course_ShadingInValvesSourceEngine.pdf
// https://developer.amd.com/wordpress/media/2013/02/Chapter1-Green-Efficient_Self-Shadowed_Radiosity_Normal_Mapping.pdf

class ExperimentRNM : public Experiment
{
public:

	struct RNMBasis
	{
		vec3 irradiance[3];

		static vec3 getVector(u32 i)
		{
			static const vec3 basisVectors[3] =
			{
				vec3(-1.0f/sqrtf(6.0f), 1.0f/sqrtf(2.0f), 1.0f/sqrtf(3.0f)),
				vec3(-1.0f/sqrtf(6.0f), -1.0f/sqrtf(2.0f), 1.0f/sqrtf(3.0f)),
				vec3(sqrtf(2.0f/3.0f), 0.0f, 1.0f/sqrtf(3.0f))
			};
			return basisVectors[i];
		}

		vec3 evaluate(const vec3& direction) const
		{
			// Efficient Self-Shadowed Radiosity Normal Mapping, Listing 1

			vec3 result = vec3(0.0f);

			vec3 dp;
			for(u32 i=0; i<3; ++i)
			{
				dp[i] = dotMax0(direction, getVector(i));
			}

			dp *= dp;

			float sum = dp.x + dp.y + dp.z;

			for(u32 i=0; i<3; ++i)
			{
				result += dp[i] * irradiance[i];
			}

			if (sum != 0.0)
			{
				result /= sum;
			}

			return result;
		}
	};

	void run(SharedData& data) override;
};

}
