#pragma once

#include <Probulator/Experiments.h>

namespace Probulator {

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
    
    ExperimentSGBase& setNonNegativeSolve(bool state)
    {
        m_nonNegativeSolve = state;
        return *this;
    }

    void run(SharedData& data) override
    {
        generateLobes();
        solveForRadiance(data.m_radianceSamples);
        generateRadianceImage(data);
        generateIrradianceImage(data);
    }

	void getProperties(std::vector<Property>& outProperties) override
	{
		Experiment::getProperties(outProperties);
		outProperties.push_back(Property("Lobe count", reinterpret_cast<int*>(&m_lobeCount)));
		outProperties.push_back(Property("Ambient lobe enabled", &m_ambientLobeEnabled));
		outProperties.push_back(Property("Lambda", &m_lambda));
		outProperties.push_back(Property("BRDF Lambda", &m_brdfLambda));
	}

    bool m_nonNegativeSolve = false;
    bool m_ambientLobeEnabled = false;
    u32 m_lobeCount = 1;
    float m_lambda = 0.0f;
    float m_brdfLambda = 0.0f; // 0 to use a curve fit for irradiance.
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
        
        // If the BRDF lambda is greater than 0, use a SG for the BRDF.
        // Otherwise, use a curve fit.
        if (m_brdfLambda > 0.f)
        {
            m_irradianceImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
                                          {
                                              brdf.p = data.m_directionImage.at(pixelPos);
                                              vec3 sampleSg = sgBasisDot(m_lobes, brdf) / pi;
                                              pixel = vec4(sampleSg, 1.0f);
                                          });
        }
        else
        {
            m_irradianceImage.forPixels2D([&](vec4& pixel, ivec2 pixelPos)
                                          {
                                              vec3 normal = data.m_directionImage.at(pixelPos);
                                              vec3 sampleSg = sgBasisIrradianceFitted(m_lobes, normal);
                                              pixel = vec4(sampleSg, 1.0f);
                                          });
        }
        
       
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
  
// A progressive least-squares solve. Reference: http://torust.me/rendering/irradiance-caching/spherical-gaussians/2018/09/21/spherical-gaussians.html
class ExperimentSGRunningAverage : public ExperimentSGBase
{
public:
    
    void solveForRadiance(const std::vector<RadianceSample>& radianceSamples) override
    {
        const u32 lobeCount = (u32)m_lobes.size();

        std::vector<float> lobeMCSphericalIntegrals(lobeCount, 0.0f);

        std::vector<float> lobePrecomputedSphericalIntegrals(lobeCount);
        for (u64 lobeIt = 0; lobeIt < lobeCount; ++lobeIt)
        {
            lobePrecomputedSphericalIntegrals[lobeIt] = (1.f - exp(-4.f * m_lobes[lobeIt].lambda)) / (4 * m_lobes[lobeIt].lambda);
        }

        float totalSampleWeight = 0.f;

        std::vector<float> sampleLobeWeights(lobeCount);

        for (const RadianceSample& sample : radianceSamples) {
            const float sampleWeight = 1.f;
            totalSampleWeight += sampleWeight;
            float sampleWeightScale = sampleWeight / totalSampleWeight;

            vec3 currentEstimate = vec3(0.f);

            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
                float dotProduct = dot(m_lobes[lobeIt].p, sample.direction);
                float weight = exp(m_lobes[lobeIt].lambda * (dotProduct - 1.0f));
                currentEstimate += m_lobes[lobeIt].mu * weight;

                sampleLobeWeights[lobeIt] = weight;
            }

            for (u32 lobeIt = 0; lobeIt < lobeCount; ++lobeIt) {
                float weight = sampleLobeWeights[lobeIt];
                if (weight == 0.f) { continue; }

                float sphericalIntegralGuess = weight * weight;
                
                // Update the MC-computed integral of the lobe over the domain.
                lobeMCSphericalIntegrals[lobeIt] += (sphericalIntegralGuess - lobeMCSphericalIntegrals[lobeIt]) * sampleWeightScale;
                
                // The most accurate method requires using the MC-computed integral,
                // since then bias in the estimate will partially cancel out.
                // However, if you don't want to store a weight per-lobe you can instead substitute it with the
                // precomputed integral at a slight increase in error.
                
                // Clamp the MC-computed integral to within a reasonable ad-hoc factor of the actual integral to avoid noise.
                float sphericalIntegral = max(lobeMCSphericalIntegrals[lobeIt], lobePrecomputedSphericalIntegrals[lobeIt] * 0.75f);
                
                vec3 otherLobesContribution = currentEstimate - m_lobes[lobeIt].mu * weight;
                vec3 newValue = (sample.value - otherLobesContribution) * weight / sphericalIntegral;
                
                m_lobes[lobeIt].mu += (newValue - m_lobes[lobeIt].mu) * sampleWeightScale;

                if (m_nonNegativeSolve) {
                    m_lobes[lobeIt].mu = max(m_lobes[lobeIt].mu, vec3(0.f));
                }
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

	void getProperties(std::vector<Property>& outProperties) override
	{
		ExperimentSGBase::getProperties(outProperties);
		outProperties.push_back(Property("Population count", reinterpret_cast<int*>(&m_populationCount)));
		outProperties.push_back(Property("Generation count", reinterpret_cast<int*>(&m_generationCount)));
	}
};

}