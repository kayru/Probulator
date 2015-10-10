#include "SGFitGeneticAlgorithm.h"
#include "Thread.h"
#include "Compute.h"

#include <random>
#include <algorithm>
#include <assert.h>

static const char* g_computeCode =
R"(
#define vec3 float3

typedef struct
{
	float pX, pY, pZ; // lobe axis
	float lambda; // sharpness
	float muX, muY, muZ; // amplitude
} SphericalGaussian;

typedef struct
{
	float dX, dY, dZ;
	float r, g, b;
} RadianceSample;

float sgEvaluateEx(vec3 p, float lambda, vec3 v)
{
	float dp = dot(v, p);
	return exp(lambda * (dp - 1.0));
}

vec3 sgEvaluate(SphericalGaussian sg, vec3 v)
{
	vec3 mu = {sg.muX, sg.muY, sg.muZ};
	vec3 p = {sg.pX, sg.pY, sg.pZ};
	return mu * sgEvaluateEx(p, sg.lambda, v);
}

__kernel void computeErrors(
	__global SphericalGaussian* lobes,
	const int lobeCount, // lobes per basis
	const int basisCount, // how many bases in total
	__global RadianceSample* radianceSamples,
	const int radianceSampleCount,
	__global float* basisErrors)
{
	int basisIt = get_global_id(0);
	float errorSquaredSum = 0.0;
	vec3 rgbLuminance = { 0.2126, 0.7152, 0.0722 };
	for (int sampleIt=0; sampleIt<radianceSampleCount; ++sampleIt)
	{
		RadianceSample sample = radianceSamples[sampleIt];
		vec3 reconstructedValue = 0.0;
		for (int lobeIt=0; lobeIt<lobeCount; ++lobeIt)
		{			
			SphericalGaussian lobe = lobes[lobeCount*basisIt + lobeIt];
			vec3 direction = {sample.dX, sample.dY, sample.dZ};
			reconstructedValue += sgEvaluate(lobe, direction);
		}
		vec3 sampleValue = {sample.r, sample.g, sample.b};
		vec3 error = sampleValue - reconstructedValue;
		errorSquaredSum += dot(error*error, rgbLuminance);
	}
	
	float mse = errorSquaredSum / radianceSampleCount;
	basisErrors[basisIt] = mse;
}
)";

namespace Probulator
{
	inline float randomFloat(std::mt19937& rng, float min = 0.0f, float max = 1.0f)
	{
		std::uniform_real_distribution<float> uniformDistribution(min, max);
		return uniformDistribution(rng);
	}

	inline u32 randomUint(std::mt19937& rng, u32 min = 0, u32 max = ~0u)
	{
		std::uniform_int_distribution<u32> uniformDistribution(min, max);
		return uniformDistribution(rng);
	}

	static float errorFunction(
		const SgBasis& basis,
		const std::vector<RadianceSample>& samples)
	{
		vec3 mse = sgBasisMeanSquareError(basis, samples);
		vec3 rgbLuminance = vec3(0.2126f, 0.7152f, 0.0722f);
		return dot(mse, rgbLuminance);
	}

	static void mutate(SgBasis& basis, float mutationRate, float sigma, std::mt19937& rng)
	{
		std::normal_distribution<float> normalDistribution(0.0f, sigma);
		for (SphericalGaussian& lobe : basis)
		{
			for (u32 channelIt = 0; channelIt <= 2; ++channelIt)
			{
				if (randomFloat(rng) <= mutationRate)
				{
					lobe.mu[channelIt] += normalDistribution(rng);
				}
			}

			if (randomFloat(rng) <= mutationRate)
			{
				lobe.lambda += normalDistribution(rng);
			}
		}
	}

	static SgBasis crossOver(const SgBasis& a, const SgBasis& b, std::mt19937& rng)
	{
		u32 crossoverPoint = randomUint(rng, 0, (u32)a.size());

		SgBasis result;

		for (u64 lobeIt = 0; lobeIt < crossoverPoint; ++lobeIt)
		{
			result.push_back(a[lobeIt]);
		}
		for (u64 lobeIt = crossoverPoint; lobeIt < a.size(); ++lobeIt)
		{
			result.push_back(b[lobeIt]);
		}

		return result;
	}

	SgBasis sgFitGeneticAlgorithm(
		const SgBasis& basis,
		const std::vector<RadianceSample>& samples,
		u32 populationCount,
		u32 generationCount,
		u32 seed,
		bool verbose)
	{
		ComputeKernel errorKernel(g_computeCode, "computeErrors");

		std::mt19937 rng(seed);

		std::vector<SgBasis> population;
		std::vector<SgBasis> nextPopulation;

		const float mutationRate = 0.05f;
		const float mutationSigma = 0.025f;
		const u32 eliteCount = 1;

		population.reserve(populationCount);
		nextPopulation.reserve(populationCount);

		for (u32 populationIt = 0; populationIt < populationCount; ++populationIt)
		{
			nextPopulation.push_back(basis);
		}

		double bestFitness = 0.0f;

		std::vector<float> populationError(populationCount);
		std::vector<double> populationFitness(populationCount);
		std::vector<u32> sortedSolutionIndices(populationCount);

		cl_int clError = CL_SUCCESS;
		cl_mem populationBuffer = clCreateBuffer(
			g_computeContext, CL_MEM_READ_ONLY,
			sizeof(SphericalGaussian)*populationCount*basis.size(),
			nullptr, &clError);
		assert(clError == CL_SUCCESS);

		cl_mem radianceBuffer = clCreateBuffer(
			g_computeContext, CL_MEM_READ_ONLY,
			sizeof(RadianceSample)*samples.size(),
			nullptr, &clError);
		assert(clError == CL_SUCCESS);

		clError = clEnqueueWriteBuffer(g_computeQueue, radianceBuffer, CL_FALSE, 0,
			sizeof(RadianceSample)*samples.size(), samples.data(),
			0, nullptr, nullptr);
		assert(clError == CL_SUCCESS);

		cl_mem errorBuffer = clCreateBuffer(
			g_computeContext, CL_MEM_WRITE_ONLY,
			sizeof(float)*populationCount,
			nullptr, &clError);
		assert(clError == CL_SUCCESS);

		for (u32 generationIt = 0; generationIt < generationCount; generationIt++)
		{
			std::swap(population, nextPopulation);

#if 0
			parallelFor(0u, populationCount, [&](u32 basisIt)
			{
				populationError[basisIt] = errorFunction(population[basisIt], samples);
			});
#else
			for (u32 basisIt = 0; basisIt < population.size(); ++basisIt)
			{
				size_t writeSize = sizeof(SphericalGaussian)*basis.size();
				size_t writeOffset = writeSize*basisIt;
				clError = clEnqueueWriteBuffer(g_computeQueue, populationBuffer, CL_FALSE,
					writeOffset, writeSize, population[basisIt].data(),
					0, nullptr, nullptr);
				assert(clError == CL_SUCCESS);
			}

			const int lobeCount = (int)basis.size();
			const int basisCount = (int)population.size();
			const int radianceSampleCount = (int)samples.size();
			clSetKernelArg(errorKernel.getKernel(), 0, sizeof(populationBuffer), &populationBuffer);
			clSetKernelArg(errorKernel.getKernel(), 1, sizeof(lobeCount), &lobeCount);
			clSetKernelArg(errorKernel.getKernel(), 2, sizeof(basisCount), &basisCount);
			clSetKernelArg(errorKernel.getKernel(), 3, sizeof(radianceBuffer), &radianceBuffer);
			clSetKernelArg(errorKernel.getKernel(), 4, sizeof(radianceSampleCount), &radianceSampleCount);
			clSetKernelArg(errorKernel.getKernel(), 5, sizeof(errorBuffer), &errorBuffer);

			size_t globalSize = population.size();
			clError = clEnqueueNDRangeKernel(g_computeQueue, errorKernel.getKernel(), 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
			assert(clError == CL_SUCCESS);

			clError = clEnqueueReadBuffer(g_computeQueue, errorBuffer, CL_FALSE, 0,
				sizeof(float)*populationCount, populationError.data(),
				0, nullptr, nullptr);
			assert(clError == CL_SUCCESS);
			clFinish(g_computeQueue);
#endif

			for (u32 populationIt = 0; populationIt < populationCount; ++populationIt)
			{
				sortedSolutionIndices[populationIt] = populationIt;
			}

			std::sort(sortedSolutionIndices.begin(), sortedSolutionIndices.end(), [&](u32 a, u32 b)
			{
				return populationError[a] < populationError[b];
			});

			u32 minErrorSolutionIndex = sortedSolutionIndices.front();
			u32 maxErrorSolutionIndex = sortedSolutionIndices.back();

			float minError = populationError[minErrorSolutionIndex];
			float maxError = populationError[maxErrorSolutionIndex];

			if (minError == 0.0f)
			{
				return population[minErrorSolutionIndex];
			}

			for (u32 populationIt = 0; populationIt < populationCount; ++populationIt)
			{
				float x = (maxError - populationError[populationIt]) / maxError;
				populationFitness[populationIt] = 0.000001 + x * populationCount;
			}

			if (verbose)
			{
				printf("Generation %d best solution error: %f\n", generationIt, minError);
			}

			nextPopulation.clear();
			for (u32 eliteIt = 0; eliteIt < eliteCount; ++eliteIt)
			{
				nextPopulation.push_back(population[sortedSolutionIndices[eliteIt]]);
			}

#ifdef _MSC_VER
			std::initializer_list<double> il(populationFitness.data(), populationFitness.data() + populationFitness.size());
			std::discrete_distribution<u32> dd(il);
#else
			std::discrete_distribution<u32> dd(populationFitness.begin(), populationFitness.end());
#endif

			while (nextPopulation.size() < populationCount)
			{
				u32 idA = dd(rng);
				u32 idB = dd(rng);
				const SgBasis& a = population[idA];
				const SgBasis& b = population[idB];
				SgBasis nextBasis = crossOver(a, b, rng);
				mutate(nextBasis, mutationRate, mutationSigma, rng);
				nextPopulation.push_back(nextBasis);
			}
		}

		clFinish(g_computeQueue);
		clReleaseMemObject(populationBuffer);
		clReleaseMemObject(radianceBuffer);
		clReleaseMemObject(errorBuffer);

		return population[sortedSolutionIndices.front()];
	}

}