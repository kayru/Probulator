#pragma once

#include "SGBasis.h"
#include "RadianceSample.h"

namespace Probulator
{
	SgBasis sgFitLeastSquares(
		const SgBasis& basis,
		const std::vector<RadianceSample>& samples);
	SgBasis sgFitNNLeastSquares(
		const SgBasis& basis,
		const std::vector<RadianceSample>& samples);
}
