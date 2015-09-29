#include "SGFitLeastSquares.h"
#include <Eigen/Eigen>
#include <Eigen/nnls.h>

namespace Probulator
{
	SgBasis sgFitLeastSquares(const SgBasis& basis, const std::vector<RadianceSample>& samples)
	{
		using namespace Eigen;
		SgBasis result = basis;

		MatrixXf A;
		A.resize(samples.size(), basis.size());
		for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt)
		{
			for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
			{
				A(sampleIt, lobeIt) = sgEvaluate(basis[lobeIt].p, basis[lobeIt].lambda, samples[sampleIt].direction);
			}
		}

		for (u32 channelIt = 0; channelIt < 3; ++channelIt)
		{
			VectorXf b;
			b.resize(samples.size());
			for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt)
			{
				b[sampleIt] = samples[sampleIt].value[channelIt];
			}

			VectorXf x = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
			for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
			{
				result[lobeIt].mu[channelIt] = x[lobeIt];
			}
		}

		return result;
	}

	// Non-negative version of least squares
	SgBasis sgFitNNLeastSquares(const SgBasis& basis, const std::vector<RadianceSample>& samples)
	{
		using namespace Eigen;
		SgBasis result = basis;

		MatrixXf A;
		A.resize(samples.size(), basis.size());
		for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt)
		{
			for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
			{
				A(sampleIt, lobeIt) = sgEvaluate(basis[lobeIt].p, basis[lobeIt].lambda, samples[sampleIt].direction);
			}
		}

		NNLS<MatrixXf> nnlssolver(A);
		for (u32 channelIt = 0; channelIt < 3; ++channelIt)
		{
			VectorXf b;
			b.resize(samples.size());
			for (u64 sampleIt = 0; sampleIt < samples.size(); ++sampleIt)
			{
				b[sampleIt] = samples[sampleIt].value[channelIt];
			}

			// -- run the solver
			nnlssolver.solve(b);
			VectorXf x = nnlssolver.x();

			for (u64 lobeIt = 0; lobeIt < basis.size(); ++lobeIt)
			{
				result[lobeIt].mu[channelIt] = x[lobeIt];
			}
		}

		return result;
	}
}
