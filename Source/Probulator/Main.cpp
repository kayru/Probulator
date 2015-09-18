#include <stdio.h>
#include "Common.h"
#include "Image.h"
#include "SphericalGaussian.h"

using namespace Probulator;

int main(int argc, char** argv)
{
#if 0

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

	auto function = [&](vec3 direction)
	{
		return (vec3)inputImage.sampleNearest(cartesianToLatLongTexcoord(direction));
	};

#else

	auto function = [&](vec3 direction)
	{
		// return 5.0f * vec3(pow(max(0.0f, -direction.z), 6.0f));
		return vec3(1.0f);
	};

#endif
	
	const ivec2 imageSize(256, 128);

	////////////////////////////////////////////
	// Generate radiance image (not convolved)
	////////////////////////////////////////////

	float averageRadianceSample = 0.0f;
	Image radianceImage(imageSize);
	radianceImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
	{
		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(imageSize - ivec2(1));
		vec3 direction = latLongTexcoordToCartesian(uv);
		vec3 sample = function(direction);
		pixel.r = sample.r;
		pixel.g = sample.g;
		pixel.b = sample.b;

		averageRadianceSample += sample.r;
	});

	radianceImage.writePng("radiance.png");

	averageRadianceSample /= radianceImage.getPixelCount();	
	printf("Average radiance: %f\n", averageRadianceSample);

	/////////////////////////////////////////
	// Project input lighting onto SG basis
	/////////////////////////////////////////

	const u32 lobeCount = 12; // <-- tweak this
	const float lambda = 0.5f * lobeCount; // <-- tweak this; 

	SphericalGaussian lobes[lobeCount];
	std::vector<vec3> sgLobeDirections;
	for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
	{
		lobes[lobeIt].p = sampleVogelsSphere(lobeIt, lobeCount);
		lobes[lobeIt].lambda = lambda;
		lobes[lobeIt].mu = vec3(0.0f);
	}

	const float normFactor = fourPi / sgIntegral(lambda); // TODO: there is no solid basis for this right now

	const u32 sampleCount = 20000;
	for (u32 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
	{
		vec2 sampleUv = sampleHammersley(sampleIt, sampleCount);
		vec3 direction = sampleUniformSphere(sampleUv);

		vec3 sample = (vec3)function(direction);

		for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
		{
			const SphericalGaussian& sg = lobes[lobeIt];
			float w = sgEvaluate(sg.p, sg.lambda, direction);
			lobes[lobeIt].mu += sample * normFactor * (w / sampleCount);
		}
	}

	/////////////////////////////////////////////////////////////////////
	// Generate irradiance image by convolving SG lighting with SG BRDF
	/////////////////////////////////////////////////////////////////////

	float averageSgIrradianceSample = 0.0f;
	Image irradianceSgImage(imageSize);
	irradianceSgImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
	{
		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(irradianceSgImage.getSize() - ivec2(1));
		vec3 direction = latLongTexcoordToCartesian(uv);

		SphericalGaussian cosineLobe = sgCosineLobe(direction);
		cosineLobe.lambda = 1.5f*cosineLobe.lambda;
		cosineLobe.mu = vec3(sgFindMu(cosineLobe.lambda, sgCosineLambda(), sgCosineMu()));

		vec3 sample = vec3(0.0f);
		for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
		{
			const SphericalGaussian& sg = lobes[lobeIt];
			sample += sgDot(sg, cosineLobe);
		}

		sample /= pi;

		pixel.r = sample.r;
		pixel.g = sample.g;
		pixel.b = sample.b;

		averageSgIrradianceSample += sample.r;
	});
	averageSgIrradianceSample /= irradianceSgImage.getPixelCount();
	printf("Average SG irradiance: %f\n", averageSgIrradianceSample);

	/////////////////////////////////////////////////////////
	// Generate reference convolved image using Monte Carlo
	/////////////////////////////////////////////////////////

	Concurrency::combinable<float> mcIrradianceSampleAccumulator;
	Image irradianceMcImage(imageSize);
	irradianceMcImage.parallelForPixels2D([&](vec4& pixel, ivec2 pixelPos)
	{
		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(imageSize - ivec2(1));
		vec3 direction = latLongTexcoordToCartesian(uv);

		mat3 basis = makeOrthogonalBasis(direction);

		vec3 sample = vec3(0.0f);
		const u32 mcSampleCount = 5000;
		for (u32 sampleIt = 0; sampleIt < mcSampleCount; ++sampleIt)
		{
			vec2 sampleUv = sampleHammersley(sampleIt, mcSampleCount);
			vec3 hemisphereDirection = sampleCosineHemisphere(sampleUv);
			vec3 sampleDirection = basis * hemisphereDirection;
			sample += function(sampleDirection);
		}

		sample /= mcSampleCount;

		pixel.r = sample.r;
		pixel.g = sample.g;
		pixel.b = sample.b;

		mcIrradianceSampleAccumulator.local() += sample.r;
	});

	irradianceMcImage.writePng("irradianceMC.png");

	float averageMcIrradianceSample = mcIrradianceSampleAccumulator.combine([](float a, float b){return a + b;});
	averageMcIrradianceSample /= irradianceMcImage.getPixelCount();
	printf("Average MC irradiance: %f\n", averageMcIrradianceSample);

	////////////////////////////////////////////////
	// Write all images into a single combined PNG
	////////////////////////////////////////////////

	Image combinedImage(imageSize.x, imageSize.y*3);
	combinedImage.paste(radianceImage, ivec2(0, imageSize.y*0));
	combinedImage.paste(irradianceSgImage, ivec2(0, imageSize.y*1));
	combinedImage.paste(irradianceMcImage, ivec2(0, imageSize.y*2));
	combinedImage.writePng("combined.png");

	return 0;
}

