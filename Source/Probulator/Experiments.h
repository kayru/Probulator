#pragma once

#include <Probulator/Common.h>
#include <Probulator/Math.h>
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
#include <Probulator/DiscreteDistribution.h>

namespace Probulator
{

class Experiment
{
public:

    class SharedData
    {
    private:

        void GenerateSamples(u32 sampleCount, Image& image, std::vector<RadianceSample>& samples)
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
			, m_sampleCount(sampleCount)
        {
            if (!m_radianceImage.readHdr(hdrFilename))
            {
                return;
            }
			
			initialize();
        }

		SharedData(u32 sampleCount, ivec2 outputSize, const Image& radianceImage)
			: m_directionImage(outputSize)
			, m_outputSize(outputSize)
			, m_sampleCount(sampleCount)
			, m_radianceImage(radianceImage)

		{
			initialize();
		}

		void initialize()
		{
			if (m_radianceImage.getSize() != m_outputSize)
			{
				m_radianceImage = imageResize(m_radianceImage, m_outputSize);
			}

			m_directionImage.forPixels2D([&](vec3& direction, ivec2 pixelPos)
			{
				vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(m_outputSize - ivec2(1));
				direction = latLongTexcoordToCartesian(uv);
			});

			GenerateSamples(m_sampleCount, m_radianceImage, m_radianceSamples);
		}

        bool isValid() const
        {
            return m_radianceImage.getSizeBytes() != 0;
        }

        void GenerateIrradianceSamples(Image& irradianceimage)
        {
            GenerateSamples(m_sampleCount, irradianceimage, m_irradianceSamples);
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

	void reset()
	{
		m_executed = false;
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
            u32 seed = m_jitterEnabled ? pixelIndex : 0;
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

        data.GenerateIrradianceSamples(m_irradianceImage);
    }

    ExperimentMCIS& setSampleCount(u32 v)
    {
        m_sampleCount = v;
        return *this;
    }

    ExperimentMCIS& setJitterEnabled(bool state)
    {
        m_jitterEnabled = state;
        return *this;
    }

    u32 m_sampleCount = 1000;
    bool m_jitterEnabled = false;
};

template <size_t L>
class ExperimentSH : public Experiment
{
public:

    void run(SharedData& data) override
    {
        SphericalHarmonicsT<vec3, L> shRadiance = {};
        const u32 sampleCount = (u32)data.m_radianceSamples.size();
        for (const RadianceSample& sample : data.m_radianceSamples)
        {
            shAddWeighted(shRadiance, shEvaluate<L>(sample.direction), sample.value * (fourPi / sampleCount));
        }

        if (m_lambda != 0.0f)
        {
            shReduceRinging<vec3, L>(shRadiance, m_lambda);
        }

        m_radianceImage = Image(data.m_outputSize);
        m_irradianceImage = Image(data.m_outputSize);

        data.m_directionImage.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
        {
            SphericalHarmonicsT<float, L> directionSh = shEvaluate<L>(direction);

            vec3 sampleSh = max(vec3(0.0f), shDot(shRadiance, directionSh));
            m_radianceImage.at(pixelPos) = vec4(sampleSh, 1.0f);

            vec3 sampleIrradianceSh = max(vec3(0.0f), shEvaluateDiffuse<vec3, L>(shRadiance, direction) / pi);
            m_irradianceImage.at(pixelPos) = vec4(sampleIrradianceSh, 1.0f);
        });
    }

    ExperimentSH<L>& setLambda(float v)
    {
        m_lambda = v;
        return *this;
    }

    float m_lambda = 0.0f;
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

    void run(SharedData& data) override
    {
        generateLobes();
        solveForRadiance(data.m_radianceSamples);
        generateRadianceImage(data);
        generateIrradianceImage(data);
    }

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

void addAllExperiments(ExperimentList& experiments);
void resetAllExperiments(ExperimentList& experiments);

} // namespace Probulator