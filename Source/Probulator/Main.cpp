#include <stdio.h>
#include "Common.h"
#include "Image.h"
#include "SphericalGaussian.h"
#include "SGBasis.h"
#include "SphericalHarmonics.h"
#include "Variance.h"
#include "RadianceSample.h"
#include "SGFitGeneticAlgorithm.h"
#include "SGFitLeastSquares.h"
#include "Compute.h"

using namespace Probulator;

// TODO: implement computeEnvmapAverage() for lat-long envmaps, taking pixel weights into account

static vec4 computeAverage(Image& image)
{
	vec4 sum = vec4(0.0f);
	image.forPixels([&](vec4& pixel){ sum += pixel; });
	sum /= image.getPixelCount();
	return sum;
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
		return (vec3)inputImage.sampleNearest(cartesianToLatLongTexcoord(direction));
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

	if (inputImage.getSize() != outputImageSize)
	{
		inputImage = imageResize(inputImage, outputImageSize);
	}

	inputImage.writeHdr("input.hdr");

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

	std::vector<RadianceSample> envmapSamples;
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

	SgBasis lobesGa = sgFitGeneticAlgorithm(lobes, envmapSamples, 50, 2000, 0, true);
	SgBasis lobesLs = sgFitLeastSquares(lobes, envmapSamples);
	SgBasis lobesNNLs = sgFitNNLeastSquares(lobes, envmapSamples);

	vec3 mseAdhoc = sgBasisMeanSquareError(lobes, envmapSamples);
	printf("Ad-hoc basis MSE: %f\n", dot(mseAdhoc, vec3(1.0f / 3.0f)));

	vec3 mseGa = sgBasisMeanSquareError(lobesGa, envmapSamples);
	printf("GA basis MSE: %f\n", dot(mseGa, vec3(1.0f / 3.0f)));

	vec3 mseLs = sgBasisMeanSquareError(lobesLs, envmapSamples);
	printf("LS basis MSE: %f\n", dot(mseLs, vec3(1.0f / 3.0f)));

	vec3 mseNNLs = sgBasisMeanSquareError(lobesNNLs, envmapSamples);
	printf("NNLS basis MSE: %f\n", dot(mseNNLs, vec3(1.0f / 3.0f)));

	///////////////////////////////////////////
	// Generate reconstructed radiance images
	///////////////////////////////////////////

	Image radianceSgImage(outputImageSize);
	Image radianceSgLsImage(outputImageSize);
	Image radianceSgNNLsImage(outputImageSize);
	Image radianceSgGaImage(outputImageSize);
	Image radianceShImage(outputImageSize);

	radianceSgImage.forPixels2D([&](vec4&, ivec2 pixelPos)
	{
		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(outputImageSize - ivec2(1));
		vec3 direction = latLongTexcoordToCartesian(uv);

		vec3 sampleSg = sgBasisEvaluate(lobes, direction);
		vec3 sampleSgGa = sgBasisEvaluate(lobesGa, direction);
		vec3 sampleSgLs = sgBasisEvaluate(lobesLs, direction);
		vec3 sampleSgNNLs = sgBasisEvaluate(lobesNNLs, direction);

		SphericalHarmonicsL2 directionSh = shEvaluateL2(direction);
		vec3 sampleSh = max(vec3(0.0f), shDot(shRadiance, directionSh));

		radianceSgLsImage.at(pixelPos) = vec4(sampleSgLs, 1.0f);
		radianceSgNNLsImage.at(pixelPos) = vec4(sampleSgNNLs, 1.0f);
		radianceSgGaImage.at(pixelPos) = vec4(sampleSgGa, 1.0f);
		radianceSgImage.at(pixelPos) = vec4(sampleSg, 1.0f);
		radianceShImage.at(pixelPos) = vec4(sampleSh, 1.0f);
	});

	radianceSgLsImage.writePng("radianceSGLS.png");
	radianceSgNNLsImage.writePng("radianceSGNNLS.png");
	radianceSgGaImage.writePng("radianceSGGA.png");
	radianceSgImage.writePng("radianceSG.png");
	radianceShImage.writePng("radianceSH.png");

	printf("Average SGLS radiance: %f\n", computeAverage(radianceSgLsImage).r);
	printf("Average SGNNLS radiance: %f\n", computeAverage(radianceSgNNLsImage).r);
	printf("Average SGGA radiance: %f\n", computeAverage(radianceSgGaImage).r);
	printf("Average SG radiance: %f\n", computeAverage(radianceSgImage).r);
	printf("Average SH radiance: %f\n", computeAverage(radianceShImage).r);

	///////////////////////////////////////////////////////////////
	// Generate irradiance image by convolving lighting with BRDF
	///////////////////////////////////////////////////////////////

	Image irradianceSgGaImage(outputImageSize);
	Image irradianceSgLsImage(outputImageSize);
	Image irradianceSgNNLsImage(outputImageSize);
	Image irradianceSgImage(outputImageSize);
	Image irradianceShImage(outputImageSize);

	//SphericalGaussian brdf = sgCosineLobe();
	SphericalGaussian brdf;
	brdf.lambda = 8.5f; // Chosen arbitrarily through experimentation
	brdf.mu = vec3(sgFindMu(brdf.lambda, pi));

	SphericalGaussian brdfGa;
	brdfGa.lambda = 3.0f; // Chosen arbitrarily through experimentation
	brdfGa.mu = vec3(sgFindMu(brdfGa.lambda, pi));

	SphericalGaussian brdfLs;
	brdfLs.lambda = 3.0f; // Chosen arbitrarily through experimentation
	brdfLs.mu = vec3(sgFindMu(brdfLs.lambda, pi));

	SphericalGaussian brdfNNLs;
	brdfNNLs.lambda = 3.0f; // Chosen arbitrarily through experimentation
	brdfNNLs.mu = vec3(sgFindMu(brdfNNLs.lambda, pi));

	irradianceSgImage.forPixels2D([&](vec4&, ivec2 pixelPos)
	{
		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(irradianceSgImage.getSize() - ivec2(1));
		vec3 direction = latLongTexcoordToCartesian(uv);

		brdf.p = direction;
		brdfGa.p = direction;
		brdfLs.p = direction;
		brdfNNLs.p = direction;
		
		vec3 sampleSg = sgBasisDot(lobes, brdf) / pi;
		irradianceSgImage.at(pixelPos) = vec4(sampleSg, 1.0f);

		vec3 sampleSgGa = sgBasisDot(lobesGa, brdfGa) / pi;
		irradianceSgGaImage.at(pixelPos) = vec4(sampleSgGa, 1.0f);

		vec3 sampleSgLs = sgBasisDot(lobesLs, brdfLs) / pi;
		irradianceSgLsImage.at(pixelPos) = vec4(sampleSgLs, 1.0f);

		vec3 sampleSgNNLs = sgBasisDot(lobesNNLs, brdfNNLs) / pi;
		irradianceSgNNLsImage.at(pixelPos) = vec4(sampleSgNNLs, 1.0f);

		vec3 sampleSh = max(vec3(0.0f), shEvaluateDiffuseL2(shRadiance, direction) / pi);
		irradianceShImage.at(pixelPos) = vec4(sampleSh, 1.0f);
	});

	printf("Average SG irradiance: %f\n", computeAverage(irradianceSgImage).r);
	printf("Average SH irradiance: %f\n", computeAverage(irradianceShImage).r);

	irradianceSgLsImage.writePng("irradianceSGLS.png");
	irradianceSgNNLsImage.writePng("irradianceSGNNLS.png");
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

	Image combinedImage(outputImageSize.x*6, outputImageSize.y*2);

	combinedImage.paste(radianceImage, outputImageSize * ivec2(0,0));
	combinedImage.paste(radianceShImage, outputImageSize * ivec2(1, 0));
	combinedImage.paste(radianceSgGaImage, outputImageSize * ivec2(2, 0));
	combinedImage.paste(radianceSgLsImage, outputImageSize * ivec2(3, 0));
	combinedImage.paste(radianceSgNNLsImage, outputImageSize * ivec2(4, 0));
	combinedImage.paste(radianceSgImage, outputImageSize * ivec2(5, 0));

	combinedImage.paste(irradianceMcImage, outputImageSize * ivec2(0, 1));
	combinedImage.paste(irradianceShImage, outputImageSize * ivec2(1, 1));
	combinedImage.paste(irradianceSgGaImage, outputImageSize * ivec2(2, 1));
	combinedImage.paste(irradianceSgLsImage, outputImageSize * ivec2(3, 1));
	combinedImage.paste(irradianceSgNNLsImage, outputImageSize * ivec2(4, 1));
	combinedImage.paste(irradianceSgImage, outputImageSize * ivec2(5, 1));
	
	combinedImage.writePng("combined.png");

	return 0;
}