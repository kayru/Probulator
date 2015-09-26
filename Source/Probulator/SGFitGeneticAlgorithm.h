#pragma once

#include "SGBasis.h"
#include "RadianceSample.h"

namespace Probulator
{
	SgBasis sgFitGeneticAlgorithm(
		const SgBasis& basis,
		const std::vector<RadianceSample>& samples,
		float lambda,
		u32 populationCount,
		u32 generationCount,
		u32 seed = 0,
		bool verbose = false);
}