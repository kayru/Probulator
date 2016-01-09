#include <Probulator/Common.h>
#include <Probulator/Image.h>
#include <Probulator/SphericalGaussian.h>
#include <Probulator/SGBasis.h>
#include <Probulator/HBasis.h>
#include <Probulator/SphericalHarmonics.h>
#include <Probulator/Variance.h>
#include <Probulator/RadianceSample.h>
#include <Probulator/SGFitGeneticAlgorithm.h>
#include <Probulator/SGFitLeastSquares.h>
#include <Probulator/Compute.h>
#include <Probulator/DiscreteDistribution.h>

#include <stdio.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <string.h>

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

using namespace Probulator;

class Experiment
{
public:

	class SharedData
	{
	private:

		void generateUniformSphereSamples(const Image& image, u32 sampleCount, std::vector<RadianceSample>& samples)
		{
			samples.reserve(sampleCount);
			for (u32 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
			{
				vec2 sampleUv = sampleHammersley(sampleIt, sampleCount);
				vec3 direction = sampleUniformSphere(sampleUv);

				vec3 sample = (vec3)image.sampleNearest(cartesianToLatLongTexcoord(direction));

				samples.push_back({ direction, sample });
			}
		}

	public:

		SharedData(u32 sampleCount, ivec2 outputSize, const char* hdrFilename)
			: m_directionImage(outputSize)
			, m_outputSize(outputSize)
		{
			m_sampleCount = sampleCount;

			if (!m_radianceImage.readHdr(hdrFilename))
			{
				return;
			}

			if (m_radianceImage.getSize() != outputSize)
			{
				m_radianceImage = imageResize(m_radianceImage, outputSize);
			}

			m_directionImage.forPixels2D([outputSize](vec3& direction, ivec2 pixelPos)
			{
				vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(outputSize - ivec2(1));
				direction = latLongTexcoordToCartesian(uv);
			});

			generateUniformSphereSamples(m_radianceImage, sampleCount, m_radianceSamples);
		}

		bool isValid() const
		{
			return m_radianceImage.getSizeBytes() != 0;
		}

		void generateIrradianceSamples(Image& irradianceimage)
		{
			generateUniformSphereSamples(irradianceimage, m_sampleCount, m_irradianceSamples);
		}

		// directions corresponding to lat-long texels
		ImageBase<vec3> m_directionImage;

		// lat-long radiance 
		Image m_radianceImage;

		// radiance samples uniformly distributed over a sphere
		std::vector<RadianceSample> m_radianceSamples;
		std::vector<RadianceSample> m_irradianceSamples;

		ivec2 m_outputSize;
		u32 m_sampleCount;
	};

	virtual void run(SharedData& data) = 0;

	virtual ~Experiment() {};

	void runWithDepencencies(SharedData& data)
	{
		if (m_executed)
			return;

		for (Experiment* d : m_dependencies)
		{
			d->runWithDepencencies(data);
		}

		run(data);

		m_executed = true;
	}

	Experiment& setEnabled(bool state)
	{
		m_enabled = state;
		return *this;
	}

	// This experiment is used as the ground truth
	Experiment& setUseAsReference(bool state)
	{
		m_useAsReference = state;
		return *this;
	}

	// This experiment requires ground truth reference as input
	Experiment& addDependency(Experiment* e)
	{
		m_dependencies.push_back(e);
		return *this;
	}

	Experiment& setInput(Experiment* e)
	{
		m_input = e;
		if (e != nullptr)
		{
			addDependency(e);
		}
		return *this;
	}

	// Experiment metadata

	std::string m_name;
	std::string m_suffix;
	bool m_executed = false;
	bool m_enabled = true;
	bool m_useAsReference = false;
	std::vector<Experiment*> m_dependencies;
	Experiment* m_input = nullptr;

	// Common experiment outputs

	Image m_radianceImage;
	Image m_irradianceImage;
};

typedef std::vector<std::unique_ptr<Experiment>> ExperimentList;

class ExperimentMC : public Experiment
{
public:

	void run(SharedData& data) override
	{
		const ivec2 imageSize = data.m_outputSize;
		const ivec2 imageSizeMinusOne = imageSize - 1;
		const vec2 imageSizeMinusOneRcp = vec2(1.0) / vec2(imageSizeMinusOne);

		m_radianceImage = data.m_radianceImage;

		m_irradianceImage = Image(data.m_outputSize);
		m_irradianceImage.parallelForPixels2D([&](vec4& pixel, ivec2 pixelPos)
		{
			vec2 uv = (vec2(pixelPos) + vec2(0.5f)) * imageSizeMinusOneRcp;
			vec3 direction = latLongTexcoordToCartesian(uv);
			mat3 basis = makeOrthogonalBasis(direction);
			vec3 accum = vec3(0.0f);
			for (u32 sampleIt = 0; sampleIt < m_hemisphereSampleCount; ++sampleIt)
			{
				vec2 sampleUv = sampleHammersley(sampleIt, m_hemisphereSampleCount);
				vec3 hemisphereDirection = sampleCosineHemisphere(sampleUv);
				vec3 sampleDirection = basis * hemisphereDirection;
				accum += (vec3)m_radianceImage.sampleNearest(cartesianToLatLongTexcoord(sampleDirection));
			}

			accum /= m_hemisphereSampleCount;

			pixel = vec4(accum, 1.0f);
		});
	}

	ExperimentMC& setHemisphereSampleCount(u32 v) { m_hemisphereSampleCount = v; return *this; }

	u32 m_hemisphereSampleCount = 1000;
};

class ExperimentMCIS : public Experiment
{
public:

	void run(SharedData& data) override
	{
		const ivec2 imageSize = data.m_outputSize;
		const ivec2 imageSizeMinusOne = imageSize - 1;
		const vec2 imageSizeMinusOneRcp = vec2(1.0) / vec2(imageSizeMinusOne);

		m_radianceImage = data.m_radianceImage;

		std::vector<float> texelWeights;
		std::vector<float> texelAreas;
		float weightSum = 0.0;
		m_radianceImage.forPixels2D([&](const vec4& p, ivec2 pixelPos)
		{
			float area = latLongTexelArea(pixelPos, imageSize);

			float intensity = dot(vec3(1.0f / 3.0f), (vec3)p);
			float weight = intensity * area;

			weightSum += weight;
			texelWeights.push_back(weight);
			texelAreas.push_back(area);
		});

		DiscreteDistribution<float> discreteDistribution(texelWeights.data(), texelWeights.size(), weightSum);

		m_irradianceImage = Image(data.m_outputSize);
		m_irradianceImage.parallelForPixels2D([&](vec4& pixel, ivec2 pixelPos)
		{
			u32 pixelIndex = pixelPos.x + pixelPos.y * m_irradianceImage.getWidth();
			u32 seed = m_scramblingEnabled ? pixelIndex : 0;
			std::mt19937 rng(seed);

			vec2 uv = (vec2(pixelPos) + vec2(0.5f)) * imageSizeMinusOneRcp;
			vec3 normal = latLongTexcoordToCartesian(uv);
			vec3 accum = vec3(0.0f);
			for (u32 sampleIt = 0; sampleIt < m_sampleCount; ++sampleIt)
			{
				u32 sampleIndex = (u32)discreteDistribution(rng);
				float sampleProbability = texelWeights[sampleIndex] / weightSum;
				vec3 sampleDirection = data.m_directionImage.at(sampleIndex);
				float cosTerm = dotMax0(normal, sampleDirection);
				float sampleArea = (float)texelAreas[sampleIndex];
				vec3 sampleRadiance = (vec3)m_radianceImage.at(sampleIndex) * sampleArea;
				accum += sampleRadiance * cosTerm / sampleProbability;
			}

			accum /= m_sampleCount * pi;

			pixel = vec4(accum, 1.0f);
		});

		data.generateIrradianceSamples(m_irradianceImage); // TODO: store irradiance samples in the MCIS experiment itself
	}

	ExperimentMCIS& setSampleCount(u32 v)
	{
		m_sampleCount = v;
		return *this;
	}

	ExperimentMCIS& setScramblingEnabled(bool state)
	{
		m_scramblingEnabled = state;
		return *this;
	}

	u32 m_sampleCount = 1000;
	bool m_scramblingEnabled = false;
};

template <size_t L>
class ExperimentSH : public Experiment
{
public:

	void run(SharedData& data) override
	{
		SphericalHarmonicsT<vec3, L> shRadiance = {};

		auto projectSh = [&](const vec4& sampleValue, ivec2 pixelPos)
		{
			const vec3& sampleDirection = data.m_directionImage.at(pixelPos);
			float weight = latLongTexelArea(pixelPos, data.m_outputSize);
			shAddWeighted(shRadiance, shEvaluate<L>(sampleDirection), (vec3)sampleValue * weight);
		};

		if (m_preconvolved)
		{
			m_input->m_irradianceImage.forPixels2D(projectSh);
		}
		else
		{
			data.m_radianceImage.forPixels2D(projectSh);
		}

		m_radianceImage = Image(data.m_outputSize);
		m_irradianceImage = Image(data.m_outputSize);

		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			SphericalHarmonicsT<float, L> directionSh = shEvaluate<L>(direction);

			vec3 sampleSh = max(vec3(0.0f), shDot(shRadiance, directionSh));
			m_radianceImage.at(pixelPos) = vec4(sampleSh, 1.0f);

			vec3 sampleIrradianceSh;
			if (m_preconvolved)
			{
				sampleIrradianceSh = sampleSh;
			}
			else
			{
				sampleIrradianceSh = max(vec3(0.0f), shEvaluateDiffuse<vec3, L>(shRadiance, direction) / pi);
			}
			m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceSh, 1.0f);
		});
	}

	ExperimentSH<L>& setLambda(float v)
	{
		m_lambda = v;
		return *this;
	}

	ExperimentSH<L>& setPreconvolved(bool state)
	{
		m_preconvolved = state;
		return *this;
	}

	float m_lambda = 0.0f;
	float m_preconvolved = false;
};

class ExperimentSHL1Geomerics : public Experiment
{
public:

	void run(SharedData& data) override
	{
		SphericalHarmonicsL1RGB shRadiance = {};
		const u32 sampleCount = (u32)data.m_radianceSamples.size();
		for (const RadianceSample& sample : data.m_radianceSamples)
		{
			shAddWeighted(shRadiance, shEvaluateL1(sample.direction), sample.value * (fourPi / sampleCount));
		}

		m_radianceImage = Image(data.m_outputSize);
		m_irradianceImage = Image(data.m_outputSize);

		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			SphericalHarmonicsL1 directionSh = shEvaluateL1(direction);

			vec3 sampleSh = max(vec3(0.0f), shDot(shRadiance, directionSh));
			m_radianceImage.at(pixelPos) = vec4(sampleSh, 1.0f);

			vec3 sampleIrradianceSh;
			for (u32 i = 0; i < 3; ++i)
			{
				SphericalHarmonicsL1 shRadianceChannel;
				shRadianceChannel[0] = shRadiance[0][i];
				shRadianceChannel[1] = shRadiance[1][i];
				shRadianceChannel[2] = shRadiance[2][i];
				shRadianceChannel[3] = shRadiance[3][i];
				sampleIrradianceSh[i] = shEvaluateDiffuseL1Geomerics(shRadianceChannel, direction) / pi;
			}
			m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceSh, 1.0f);
		});
	}
};

template <size_t L>
class ExperimentHBasis : public Experiment
{
public:

	void run(SharedData& data) override
	{
		HBasisT<vec3, L> hRadiance = {};
		const u32 sampleCount = (u32)data.m_radianceSamples.size();
		for (const RadianceSample& sample : data.m_radianceSamples)
		{
			hAddWeighted(hRadiance, hEvaluate<L>(sample.direction), sample.value * (fourPi / sampleCount));
		}

		HBasisT<vec3, L> hIrradiance = {};
		for (const RadianceSample& sample : data.m_irradianceSamples)
		{
			hAddWeighted(hIrradiance, hEvaluate<L>(sample.direction), sample.value * (fourPi / sampleCount));
		}

		m_radianceImage = Image(data.m_outputSize);
		m_irradianceImage = Image(data.m_outputSize);

		data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
		{
			HBasisT<float, L> directionH = hEvaluate<L>(direction);

			vec3 sampleH = max(vec3(0.0f), hDot(hRadiance, directionH));
			m_radianceImage.at(pixelPos) = vec4(sampleH, 1.0f);

			vec3 sampleIrradianceH = max(vec3(0.0f), hDot(hRadiance, directionH));
			m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceH, 1.0f);
		});
	}
};

class ExperimentSGBase : public Experiment
{
public:

	ExperimentSGBase& setLobeCountAndLambda(u32 lobeCount, float lambda)
	{ 
		m_lobeCount = lobeCount;
		m_lambda = lambda;
		return *this; 
	}

	ExperimentSGBase& setBrdfLambda(float brdfLambda)
	{
		m_brdfLambda = brdfLambda;
		return *this;
	}

	ExperimentSGBase& setAmbientLobeEnabled(bool state)
	{
		m_ambientLobeEnabled = state;
		return *this;
	}

	ExperimentSGBase& setPreconvolved(bool state)
	{
		m_preconvolved = state;
		return *this;
	}

	void run(SharedData& data) override
	{
		generateLobes();

		if (m_preconvolved)
		{
			solveForRadiance(data.m_irradianceSamples);
			generateRadianceImage(data);
			m_irradianceImage = m_radianceImage;
		}
		else
		{
			solveForRadiance(data.m_radianceSamples);
			generateRadianceImage(data);
			generateIrradianceImage(data);
		}
	}

	bool m_preconvolved = false;
	bool m_ambientLobeEnabled = false;
	u32 m_lobeCount = 1;
	float m_lambda = 0.0f;
	float m_brdfLambda = 0.0f;
	SgBasis m_lobes;

protected:

	virtual void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) = 0;

	void generateLobes()
	{
		std::vector<vec3> sgLobeDirections(m_lobeCount);

		m_lobes.resize(m_lobeCount);
		for (u32 lobeIt = 0; lobeIt < m_lobeCount; ++lobeIt)
		{
			sgLobeDirections[lobeIt] = sampleVogelsSphere(lobeIt, m_lobeCount);

			m_lobes[lobeIt].p = sgLobeDirections[lobeIt];
			m_lobes[lobeIt].lambda = m_lambda;
			m_lobes[lobeIt].mu = vec3(0.0f);
		}

		if (m_ambientLobeEnabled)
		{
			SphericalGaussian lobe;
			lobe.p = vec3(0.0f, 0.0f, 1.0f);
			lobe.lambda = 0.0f; // cover entire sphere
			lobe.mu = vec3(0.0f);
			m_lobes.push_back(lobe);
		}
	}

	void generateRadianceImage(const SharedData& data)
	{
		m_radianceImage = Image(data.m_outputSize);
		m_radianceImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
		{
			vec3 direction = data.m_directionImage.at(pixelPos);
			vec3 sampleSg = sgBasisEvaluate(m_lobes, direction);
			pixel = vec4(sampleSg, 1.0f);
		});
	}

	void generateIrradianceImage(const SharedData& data)
	{
		SphericalGaussian brdf;
		brdf.lambda = m_brdfLambda;
		brdf.mu = vec3(sgFindMu(brdf.lambda, pi));

		m_irradianceImage = Image(data.m_outputSize);
		m_irradianceImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
		{
			brdf.p = data.m_directionImage.at(pixelPos);
			vec3 sampleSg = sgBasisDot(m_lobes, brdf) / pi;
			pixel = vec4(sampleSg, 1.0f);
		});
	}
};

class ExperimentSGNaive : public ExperimentSGBase
{
public:

	void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
	{
		const u32 lobeCount = (u32)m_lobes.size();
		const u32 sampleCount = (u32)radianceSamples.size();
		const float normFactor = sgBasisNormalizationFactor(m_lambda, lobeCount);

		for (const RadianceSample& sample : radianceSamples)
		{
			for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
			{
				const SphericalGaussian& sg = m_lobes[lobeIt];
				float w = sgEvaluate(sg.p, sg.lambda, sample.direction);
				m_lobes[lobeIt].mu += sample.value * normFactor * (w / sampleCount);
			}
		}
	}
};

class ExperimentSGLS : public ExperimentSGBase
{
public:

	void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
	{
		m_lobes = sgFitLeastSquares(m_lobes, radianceSamples);
	}
};

class ExperimentSGNNLS : public ExperimentSGBase
{
public:

	void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
	{
		m_lobes = sgFitNNLeastSquares(m_lobes, radianceSamples);
	}
};

class ExperimentSGGA : public ExperimentSGBase
{
public:

	u32 m_populationCount = 50;
	u32 m_generationCount = 2000;

	ExperimentSGBase& setPopulationAndGenerationCount(u32 populationCount, u32 generationCount)
	{
		m_populationCount = populationCount;
		m_generationCount = generationCount;
		return *this;
	}

	void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
	{
		m_lobes = sgFitNNLeastSquares(m_lobes, radianceSamples); // NNLS is used to seed GA
		m_lobes = sgFitGeneticAlgorithm(m_lobes, radianceSamples, m_populationCount, m_generationCount);

	}
};

void generateReportHtml(const ExperimentList& experiments, const char* filename)
{
	Experiment* referenceMode = nullptr;
	for(const auto& it : experiments)
	{
		if (it->m_useAsReference && it->m_enabled)
		{
			referenceMode = it.get();
			break;
		}
	}

	std::ofstream f;
	f.open(filename);
	f << "<!DOCTYPE html>" << std::endl;
	f << "<html>" << std::endl;
	f << "<table>" << std::endl;
	f << "<tr><td>Radiance</td><td>Irradiance</td>";
	if (referenceMode)
	{
		f << "<td>Irradiance Error (sMAPE)</td>";
	}
	f << "<td>Mode</td></tr>" << std::endl;
	for (const auto& it : experiments)
	{
		if (!it->m_enabled)
			continue;

		std::ostringstream radianceFilename;
		radianceFilename << "radiance" << it->m_suffix << ".png";
		it->m_radianceImage.writePng(radianceFilename.str().c_str());

		std::ostringstream irradianceFilename;
		irradianceFilename << "irradiance" << it->m_suffix << ".png";
		it->m_irradianceImage.writePng(irradianceFilename.str().c_str());

		f << "<tr>";

		f << "<td valign=\"top\"><img src=\"" << radianceFilename.str() << "\"/>";
		if (referenceMode && referenceMode != it.get())
		{
			f << "<br/>";
			vec4 mse = imageMeanSquareError(referenceMode->m_radianceImage, it->m_radianceImage);
			float mseScalar = dot(vec3(1.0f/3.0f), (vec3)mse);
			f << "MSE: " << mseScalar << " ";
			f << "RMS: " << sqrtf(mseScalar);
		}
		f << "</td>";

		f << "<td valign=\"top\"><img src=\"" << irradianceFilename.str() << "\"/>";
		if (referenceMode && referenceMode != it.get())
		{
			f << "<br/>";
			vec4 mse = imageMeanSquareError(referenceMode->m_irradianceImage, it->m_irradianceImage);
			float mseScalar = dot(vec3(1.0f/3.0f), (vec3)mse);
			f << "MSE: " << mseScalar << " ";
			f << "RMS: " << sqrtf(mseScalar);
		}
		f << "</td>";

		if (referenceMode)
		{
			if (referenceMode != it.get())
			{
				std::ostringstream irradianceErrorFilename;
				irradianceErrorFilename << "irradianceError" << it->m_suffix << ".png";
				it->m_radianceImage.writePng(irradianceErrorFilename.str().c_str());

				Image errorImage = imageSymmetricAbsolutePercentageError(referenceMode->m_irradianceImage, it->m_irradianceImage);
				for (vec4& pixel : errorImage)
				{
					pixel.w = 1.0f;
				}
				errorImage.writePng(irradianceErrorFilename.str().c_str());

				f << "<td valign=\"top\"><img src=\"" << irradianceErrorFilename.str() << "\"/></td>";
			}
			else
			{
				f << "<td><center><b>N/A</b></center></td>";
			}
		}

		f << "<td>" << it->m_name;
		if (referenceMode == it.get())
		{
			f << "<br><b>Reference</b>";
		}
		f << "</td>";

		f << "</tr>";
		f << std::endl;
	}
	f << "</table>" << std::endl;
	f << "</html>" << std::endl;
	f.close();
}

template <typename T> 
inline T& addExperiment(ExperimentList& list, const char* name, const char* suffix)
{
	T* e = new T;
	list.push_back(std::unique_ptr<Experiment>(e));
	e->m_name = name;
	e->m_suffix = suffix;
	return *e;
}

static void enableExperimentsBySuffix(ExperimentList& list, u32 suffixCount, char** suffixes)
{
	for (const auto& e : list)
	{
		e->m_enabled = false;
		for (u32 i = 0; i < suffixCount; ++i)
		{
			if (!strcasecmp(e->m_suffix.c_str(), suffixes[i]))
			{
				e->m_enabled = true;
				break;
			}
		}
	}
}

int main(int argc, char** argv)
{
	const ivec2 outputImageSize(256, 128);
	const u32 lobeCount = 12; // <-- tweak this
	const float lambda = 0.5f * lobeCount; // <-- tweak this; 
	const u32 sampleCount = 20000;

	if (argc < 2)
	{
		printf("Usage: Probulator <LatLongEnvmap.hdr> [enabled experiments by suffix]\n");
		return 1;
	}

	const char* inputFilename = argv[1];

	printf("Loading '%s'\n", inputFilename);

	Experiment::SharedData sharedData(sampleCount, outputImageSize, inputFilename);

	if (!sharedData.isValid())
	{
		printf("ERROR: Failed to read input image from file '%s'\n", inputFilename);
		return 1;
	}

	ExperimentList experiments;

	Experiment* experimentMCIS = &addExperiment<ExperimentMCIS>(experiments, "Monte Carlo [Importance Sampling]", "MCIS")
		.setSampleCount(5000)
		.setScramblingEnabled(false) // prefer errors due to correlation instead of noise due to scrambling
		.setUseAsReference(true); // other experiments will be compared against this

	addExperiment<ExperimentMCIS>(experiments, "Monte Carlo [Importance Sampling, Scrambled]", "MCISS")
		.setSampleCount(5000)
		.setScramblingEnabled(true)
		.setEnabled(false); // disabled by default, since MCIS mode is superior

	addExperiment<ExperimentMC>(experiments, "Monte Carlo", "MC")
		.setHemisphereSampleCount(5000)
		.setEnabled(false); // disabled by default, since MCIS mode is superior

	addExperiment<ExperimentSHL1Geomerics>(experiments, "Spherical Harmonics L1 [Geomerics]", "SHL1G");

	addExperiment<ExperimentSH<1>>(experiments, "Spherical Harmonics L1", "SHL1");

	addExperiment<ExperimentSH<1>>(experiments, "Spherical Harmonics L1 [Preconvolved]", "SHL1p")
		.setPreconvolved(true)
		.setEnabled(false)
		.setInput(experimentMCIS);

	addExperiment<ExperimentSH<2>>(experiments, "Spherical Harmonics L2", "SHL2");

	addExperiment<ExperimentSH<2>>(experiments, "Spherical Harmonics L2 [Preconvolved]", "SHL2p")
		.setPreconvolved(true)
		.setEnabled(false)
		.setInput(experimentMCIS);

	addExperiment<ExperimentSH<3>>(experiments, "Spherical Harmonics L3", "SHL3");

	addExperiment<ExperimentSH<3>>(experiments, "Spherical Harmonics L3 [Preconvolved]", "SHL3p")
		.setPreconvolved(true)
		.setEnabled(false)
		.setInput(experimentMCIS);

	addExperiment<ExperimentSH<4>>(experiments, "Spherical Harmonics L4", "SHL4");

	addExperiment<ExperimentSH<4>>(experiments, "Spherical Harmonics L4 [Preconvolved]", "SHL4p")
		.setPreconvolved(true)
		.setEnabled(false)
		.setInput(experimentMCIS);

	addExperiment<ExperimentHBasis<4>>(experiments, "HBasis-4", "H4")
		.setInput(experimentMCIS);

	addExperiment<ExperimentHBasis<6>>(experiments, "HBasis-6", "H6")
		.setInput(experimentMCIS);

	addExperiment<ExperimentSGNaive>(experiments, "Spherical Gaussians [Naive]", "SG")
		.setBrdfLambda(8.5f) // Chosen arbitrarily through experimentation
		.setLobeCountAndLambda(lobeCount, lambda);

	addExperiment<ExperimentSGNaive>(experiments, "Spherical Gaussians [Naive, Preconvolved]", "SGp")
		.setLobeCountAndLambda(lobeCount, lambda)
		.setPreconvolved(true)
		.setInput(experimentMCIS)
		.setEnabled(false);

	addExperiment<ExperimentSGLS>(experiments, "Spherical Gaussians [Least Squares]", "SGLS")
		.setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
		.setLobeCountAndLambda(lobeCount, lambda);

	addExperiment<ExperimentSGLS>(experiments, "Spherical Gaussians [Least Squares, Preconvolved]", "SGLSp")
		.setLobeCountAndLambda(lobeCount, lambda)
		.setPreconvolved(true)
		.setInput(experimentMCIS)
		.setEnabled(false);

	addExperiment<ExperimentSGLS>(experiments, "Spherical Gaussians [Least Squares + Ambient]", "SGLSA")
		.setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
		.setAmbientLobeEnabled(true)
		.setLobeCountAndLambda(lobeCount, lambda);

	addExperiment<ExperimentSGNNLS>(experiments, "Spherical Gaussians [Non-Negative Least Squares]", "SGNNLS")
		.setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
		.setLobeCountAndLambda(lobeCount, lambda);

	addExperiment<ExperimentSGNNLS>(experiments, "Spherical Gaussians [Non-Negative Least Squares, Preconvolved]", "SGNNLSp")
		.setLobeCountAndLambda(lobeCount, lambda)
		.setPreconvolved(true)
		.setInput(experimentMCIS)
		.setEnabled(false);

	addExperiment<ExperimentSGGA>(experiments, "Spherical Gaussians [Genetic Algorithm]", "SGGA")
		.setPopulationAndGenerationCount(50, 2000)
		.setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
		.setLobeCountAndLambda(lobeCount, lambda)
		.setEnabled(false); // disabled by default, as it requires *very* long time to converge

	if (argc > 2)
	{
		enableExperimentsBySuffix(experiments, argc - 2, argv + 2);
	}

	printf("Running experiments:\n");

	for (const auto& e : experiments)
	{
		if (!e->m_enabled)
			continue;

		printf("  * %s\n", e->m_name.c_str());
		e->runWithDepencencies(sharedData);
	}

	generateReportHtml(experiments, "report.html");

	return 0;
}