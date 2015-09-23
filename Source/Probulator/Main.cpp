#include <stdio.h>
#include "Common.h"
#include "Image.h"
#include "SphericalGaussian.h"
#include "SphericalHarmonics.h"
#include "Variance.h"

#include <random>
#include <algorithm>

using namespace Probulator;

// TODO: implement computeEnvmapAverage() for lat-long envmaps, taking pixel weights into account

static vec4 computeAverage(Image& image)
{
	vec4 sum = vec4(0.0f);
	image.forPixels([&](vec4& pixel){ sum += pixel; });
	sum /= image.getPixelCount();
	return sum;
}

static float sgBasisNormalizationFactor(float lambda, u32 lobeCount)
{
	// TODO: there is no solid basis for this right now
	float num = 4.0f * lambda * lambda;
	float den = (1.0f - exp(-2.0f*lambda)) * lobeCount;
	return num / den;
}

struct EnvmapSample
{
	vec3 direction;
	vec3 value;
};

vec3 computeMeanSquareError(
	const EnvmapSample* samples, u32 sampleCount,
	const SphericalGaussian* lobes, u32 lobeCount)
{
	vec3 errorSquaredSum = vec3(0.0f);

	float sampleWeight = 1.0f / sampleCount;
	for (u32 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
	{
		const vec3& direction = samples[sampleIt].direction;
		vec3 reconstructedValue = vec3(0.0f);
		for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
		{
			reconstructedValue += sgEvaluate(lobes[lobeIt], direction);
		}
		vec3 error = samples[sampleIt].value - reconstructedValue;
		errorSquaredSum += error*error;
	}

	return errorSquaredSum / vec3((float)sampleCount);
}

void computeMeanAndVariance(
	const SphericalGaussian* lobes, u32 lobeCount,
	u32 sampleCount,
	vec3& outMean, vec3& outVariance)
{
	OnlineVariance<vec3> accumulator;

	for (u32 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
	{
		vec2 sampleUv = sampleHammersley(sampleIt, sampleCount);
		vec3 direction = sampleUniformSphere(sampleUv);

		vec3 reconstructedValue = vec3(0.0f);
		for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
		{
			reconstructedValue += sgEvaluate(lobes[lobeIt], direction);
		}

		accumulator.addSample(reconstructedValue);
	}

	outMean = accumulator.mean;
	outVariance = accumulator.getVariance();
}

typedef std::vector<SphericalGaussian> SgBasis;

static float fitnessFunction(
	const SgBasis& basis,
	const std::vector<EnvmapSample>& samples,
	const vec3& channelMask)
{
	vec3 mse = computeMeanSquareError(
		samples.data(), (u32)samples.size(), 
		basis.data(), (u32)basis.size());
	float den = dot(mse, channelMask);
	return den > 0.0f ? 1.0f / den : FLT_MAX;
}

static void mutate(SgBasis& basis, std::mt19937& rng)
{
	std::uniform_real_distribution<float> uniformDistribution(-1.0f, 1.0f);
	std::uniform_int_distribution<u64> uid(0, basis.size()-1);

	u64 i = uid(rng);
	SphericalGaussian& lobe = basis[i];
	float r = uniformDistribution(rng);
	if (abs(r) < 0.01f)
	{
		lobe.mu += r;
	}
	else
	{
		lobe.mu += r * 0.1f;
	}
}

static SgBasis crossOver(const SgBasis& a, const SgBasis& b, std::mt19937& rng)
{
	std::uniform_int_distribution<u64> ud(0, a.size());
	u64 crossoverPoint = ud(rng);

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

static SgBasis geneticAlgorithm(
	const std::vector<EnvmapSample>& samples,
	const std::vector<vec3>& lobeDirections,
	float lambda,
	u32 populationCount,
	u32 generationCount,
	u32 seed = 0)
{
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> uniformDistribution;

	std::vector<double> populationFitness;
	
	std::vector<SgBasis> population;
	std::vector<SgBasis> nextPopulation;

	population.reserve(populationCount);
	nextPopulation.reserve(populationCount);

	for (u32 populationIt = 0; populationIt < populationCount; ++populationIt)
	{
		SgBasis basis;
		for (u64 lobeIt = 0; lobeIt < lobeDirections.size(); ++lobeIt)
		{
			SphericalGaussian lobe;
			lobe.p = lobeDirections[lobeIt];
			lobe.lambda = lambda;
			lobe.mu = vec3(uniformDistribution(rng));
			basis.push_back(lobe);
		}
		population.push_back(basis);
	}

	double bestFitness = 0.0f;

	populationFitness.resize(populationCount);

	for (u32 generationIt = 0; generationIt < generationCount; generationIt++)
	{
		parallelFor(0u, populationCount, [&](u32 basisIt)
		{
			const SgBasis& basis = population[basisIt];
			float fitness = fitnessFunction(basis, samples, vec3(1.0f, 0.0f, 0.0f));
			populationFitness[basisIt] = fitness;
		});

		std::vector<u32> sortedSolutionIndices(populationCount);
		sortedSolutionIndices.reserve(populationCount);
		for (u32 populationIt = 0; populationIt < populationCount; ++populationIt)
		{
			sortedSolutionIndices[populationIt] = populationIt;
		}

		std::sort(sortedSolutionIndices.begin(), sortedSolutionIndices.end(), [&](u32 a, u32 b)
		{
			return populationFitness[a] > populationFitness[b];
		});

		u32 bestSolutionIndex = sortedSolutionIndices[0];
		double bestFitness = populationFitness[bestSolutionIndex];
		printf("Generation %d best solution fitness: %f\n", generationIt, bestFitness);
		
		if (bestFitness == FLT_MAX)
		{
			return population[bestSolutionIndex];
		}

		nextPopulation.clear();
		const u32 eliteCount = 15;
		for (u32 eliteIt = 0; eliteIt < eliteCount; ++eliteIt)
		{
			nextPopulation.push_back(population[sortedSolutionIndices[eliteIt]]);
		}

		std::initializer_list<double> il(populationFitness.data(), populationFitness.data() + populationFitness.size());
		std::discrete_distribution<int> dd(il);
		while (nextPopulation.size() < populationCount)
		{
			int idA = dd(rng);
			int idB = dd(rng);
			const SgBasis& a = population[idA];
			const SgBasis& b = population[idB];
			SgBasis basis = crossOver(a, b, rng);
			mutate(basis, rng);
			nextPopulation.push_back(basis);
		}

		std::swap(population, nextPopulation);
	}

	u64 bestResultIndex = std::distance(
		populationFitness.begin(), 
		std::max_element(populationFitness.begin(), populationFitness.end()));

	const SgBasis& result = population[bestResultIndex];

	return result;
}

int main(int argc, char** argv)
{
#if 1

	if (argc < 2)
	{
		printf("Usage: Probulator <LatLongEnvmap.hdr>\n");
		return 1;
	}

	const char* inputFilename = argv[1];
	Image inputImage;
	if (!inputImage.readHdr(inputFilename))
	{
		printf("ERROR: Failed to read input image from file '%s'\n", inputFilename);
		return 1;
	}

	auto getSample = [&](vec3 direction)
	{
		return (vec3)inputImage.sampleNearest(cartesianToLatLongTexcoord(direction)).r;
	};

#else

	auto getSample = [&](vec3 direction)
	{
		//return vec3(max(0.0f, -direction.z));
		return 10.0f * vec3(pow(max(0.0f, -direction.z), 100.0f));
		//return 200.0f * vec3(pow(max(0.0f, -direction.z), 100.0f));
		//return vec3(0.5f);
	};

#endif

	const ivec2 outputImageSize(256, 128);

	//////////////////////
	// Generate SG basis
	//////////////////////

	const u32 lobeCount = 12; // <-- tweak this
	const float lambda = 0.5f * lobeCount; // <-- tweak this; 

	SgBasis lobes(lobeCount);
	std::vector<vec3> sgLobeDirections;
	for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
	{
		sgLobeDirections.push_back(sampleVogelsSphere(lobeIt, lobeCount));

		lobes[lobeIt].p = sgLobeDirections.back();
		lobes[lobeIt].lambda = lambda;
		lobes[lobeIt].mu = vec3(0.0f);		
	}

	const float sgNormFactor = sgBasisNormalizationFactor(lambda, lobeCount);

	////////////////////////////////////////////
	// Generate radiance image (not convolved)
	////////////////////////////////////////////

	Image radianceImage(outputImageSize);
	radianceImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
	{
		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(outputImageSize - ivec2(1));
		vec3 direction = latLongTexcoordToCartesian(uv);
		vec3 sample = getSample(direction);

		pixel = vec4(sample, 1.0f);
	});

	radianceImage.writePng("radiance.png");

	printf("Average radiance: %f\n", computeAverage(radianceImage).r);

	/////////////////////
	// Project radiance
	/////////////////////

	const u32 sampleCount = 20000;

	std::vector<EnvmapSample> envmapSamples;
	envmapSamples.reserve(sampleCount);

	SphericalHarmonicsL2RGB shRadiance;

	for (u32 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
	{
		vec2 sampleUv = sampleHammersley(sampleIt, sampleCount);
		vec3 direction = sampleUniformSphere(sampleUv);

		vec3 sample = getSample(direction);

		for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
		{
			const SphericalGaussian& sg = lobes[lobeIt];
			float w = sgEvaluate(sg.p, sg.lambda, direction);
			lobes[lobeIt].mu += sample * sgNormFactor * (w / sampleCount);
		}

		shAddWeighted(shRadiance, shEvaluateL2(direction), sample * (fourPi / sampleCount));

		envmapSamples.push_back({direction, sample});
	}

	float adhocFitness = fitnessFunction(lobes, envmapSamples, vec3(1.0f, 0.0f, 0.0f));
	printf("Adhoc basis fitness: %f\n", adhocFitness);

	SgBasis optimizedBasis = geneticAlgorithm(envmapSamples, sgLobeDirections, lambda, 100, 200);

	///////////////////////////////////////////
	// Generate reconstructed radiance images
	///////////////////////////////////////////

	Image radianceSgImage(outputImageSize);
	Image radianceSgGaImage(outputImageSize);
	Image radianceShImage(outputImageSize);

	radianceSgImage.forPixels2D([&](vec4&, ivec2 pixelPos)
	{
		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(outputImageSize - ivec2(1));
		vec3 direction = latLongTexcoordToCartesian(uv);

		vec3 sampleSg = vec3(0.0f);
		for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
		{
			const SphericalGaussian& sg = lobes[lobeIt];
			sampleSg += sgEvaluate(sg, direction);
		}

		vec3 sampleSgGa = vec3(0.0f);
		for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
		{
			const SphericalGaussian& sg = optimizedBasis[lobeIt];
			sampleSgGa += sgEvaluate(sg, direction);
		}

		SphericalHarmonicsL2 directionSh = shEvaluateL2(direction);
		vec3 sampleSh = max(vec3(0.0f), shDot(shRadiance, directionSh));

		radianceSgGaImage.at(pixelPos) = vec4(sampleSgGa, 1.0f);
		radianceSgImage.at(pixelPos) = vec4(sampleSg, 1.0f);
		radianceShImage.at(pixelPos) = vec4(sampleSh, 1.0f);
	});

	radianceSgGaImage.writePng("radianceSGGA.png");
	radianceSgImage.writePng("radianceSG.png");
	radianceShImage.writePng("radianceSH.png");

	printf("Average SGGA radiance: %f\n", computeAverage(radianceSgGaImage).r);
	printf("Average SG radiance: %f\n", computeAverage(radianceSgImage).r);
	printf("Average SH radiance: %f\n", computeAverage(radianceShImage).r);

	vec3 radianceSgMse = computeMeanSquareError(envmapSamples.data(), sampleCount, lobes.data(), lobeCount);
	printf("SG radiance projection MSE: %f\n", dot(radianceSgMse, vec3(1.0f / 3.0f)));

	vec3 radianceSgMean, radianceSgVariance;
	computeMeanAndVariance(lobes.data(), lobeCount, sampleCount, radianceSgMean, radianceSgVariance);

	printf("SG radiance mean: %f\n", radianceSgMean.r);
	printf("SG radiance variance : %f\n", radianceSgVariance.r);

	///////////////////////////////////////////////////////////////
	// Generate irradiance image by convolving lighting with BRDF
	///////////////////////////////////////////////////////////////

	Image irradianceSgGaImage(outputImageSize);
	Image irradianceSgImage(outputImageSize);
	Image irradianceShImage(outputImageSize);

	//SphericalGaussian brdf = sgCosineLobe();
	SphericalGaussian brdf;
	brdf.lambda = 8.5f; // Chosen arbitrarily through experimentation
	brdf.mu = vec3(sgFindMu(brdf.lambda, pi));

	SphericalGaussian brdfGa;
	brdfGa.lambda = 3.5f; // Chosen arbitrarily through experimentation
	brdfGa.mu = vec3(sgFindMu(brdfGa.lambda, pi));

	irradianceSgImage.forPixels2D([&](vec4&, ivec2 pixelPos)
	{
		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(irradianceSgImage.getSize() - ivec2(1));
		vec3 direction = latLongTexcoordToCartesian(uv);

		brdf.p = direction;
		
		vec3 sampleSg = vec3(0.0f);
		for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
		{
			const SphericalGaussian& sg = lobes[lobeIt];
			sampleSg += sgDot(sg, brdf);
		}

		sampleSg /= pi;
		irradianceSgImage.at(pixelPos) = vec4(sampleSg, 1.0f);

		brdfGa.p = direction;

		vec3 sampleSgGa = vec3(0.0f);
		for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
		{
			const SphericalGaussian& sg = optimizedBasis[lobeIt];
			sampleSgGa += sgDot(sg, brdfGa);
		}

		sampleSgGa /= pi;
		irradianceSgGaImage.at(pixelPos) = vec4(sampleSgGa, 1.0f);

		vec3 sampleSh = max(vec3(0.0f), shEvaluateDiffuseL2(shRadiance, direction) / pi);
		irradianceShImage.at(pixelPos) = vec4(sampleSh, 1.0f);
	});

	printf("Average SG irradiance: %f\n", computeAverage(irradianceSgImage).r);
	printf("Average SH irradiance: %f\n", computeAverage(irradianceShImage).r);

	irradianceSgGaImage.writePng("irradianceSGGA.png");
	irradianceSgImage.writePng("irradianceSG.png");
	irradianceShImage.writePng("irradianceSH.png");

	/////////////////////////////////////////////////////////
	// Generate reference convolved image using Monte Carlo
	/////////////////////////////////////////////////////////

	Image irradianceMcImage(outputImageSize);
	irradianceMcImage.parallelForPixels2D([&](vec4& pixel, ivec2 pixelPos)
	{
		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(outputImageSize - ivec2(1));
		vec3 direction = latLongTexcoordToCartesian(uv);

		mat3 basis = makeOrthogonalBasis(direction);

		vec3 sample = vec3(0.0f);
		const u32 mcSampleCount = 5000;
		for (u32 sampleIt = 0; sampleIt < mcSampleCount; ++sampleIt)
		{
			vec2 sampleUv = sampleHammersley(sampleIt, mcSampleCount);
			vec3 hemisphereDirection = sampleCosineHemisphere(sampleUv);
			vec3 sampleDirection = basis * hemisphereDirection;
			sample += getSample(sampleDirection);
		}

		sample /= mcSampleCount;

		pixel = vec4(sample, 1.0f);
	});

	irradianceMcImage.writePng("irradianceMC.png");

	printf("Average MC irradiance: %f\n", computeAverage(irradianceMcImage).r);

	////////////////////////////////////////////////
	// Write all images into a single combined PNG
	////////////////////////////////////////////////

	Image combinedImage(outputImageSize.x*4, outputImageSize.y*2);

	combinedImage.paste(radianceImage, outputImageSize * ivec2(0,0));
	combinedImage.paste(radianceShImage, outputImageSize * ivec2(1, 0));
	combinedImage.paste(radianceSgGaImage, outputImageSize * ivec2(2, 0));
	combinedImage.paste(radianceSgImage, outputImageSize * ivec2(3, 0));

	combinedImage.paste(irradianceMcImage, outputImageSize * ivec2(0, 1));
	combinedImage.paste(irradianceShImage, outputImageSize * ivec2(1, 1));
	combinedImage.paste(irradianceSgGaImage, outputImageSize * ivec2(2, 1));
	combinedImage.paste(irradianceSgImage, outputImageSize * ivec2(3, 1));
	
	combinedImage.writePng("combined.png");

	return 0;
}

