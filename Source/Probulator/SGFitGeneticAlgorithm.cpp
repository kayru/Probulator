#include "SGFitGeneticAlgorithm.h"
#include "Thread.h"

#include <random>
#include <algorithm>

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
					float delta = normalDistribution(rng);
					lobe.mu[channelIt] += delta;
				}
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
		float lambda,
		u32 populationCount,
		u32 generationCount,
		u32 seed,
		bool verbose)
	{
		std::mt19937 rng(seed);

		std::vector<SgBasis> population;
		std::vector<SgBasis> nextPopulation;

		const float mutationRate = 0.05f;
		const float mutationSigma = 0.025f;
		const u32 eliteCount = 5;

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

		for (u32 generationIt = 0; generationIt < generationCount; generationIt++)
		{
			std::swap(population, nextPopulation);

			parallelFor(0u, populationCount, [&](u32 basisIt)
			{
				populationError[basisIt] = errorFunction(population[basisIt], samples);
			});

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
				float x = (maxError-populationError[populationIt]) / maxError;
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

			std::initializer_list<double> il(populationFitness.data(), populationFitness.data() + populationFitness.size());
			std::discrete_distribution<u32> dd(il);

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

		return population[sortedSolutionIndices.front()];
	}

}