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

	enum PropertyType
	{
		PropertyType_Bool,
		PropertyType_Float,
		PropertyType_Int,
		PropertyType_Vec2,
		PropertyType_Vec3,
		PropertyType_Vec4,
	};

	struct Property
	{
		Property(const char* name, bool* data) : m_name(name), m_type(PropertyType_Bool) { m_data.asBool = data; }
		Property(const char* name, float* data) : m_name(name), m_type(PropertyType_Float) { m_data.asFloat = data; }
		Property(const char* name, int* data) : m_name(name), m_type(PropertyType_Int) { m_data.asInt = data; }
		Property(const char* name, vec2* data) : m_name(name), m_type(PropertyType_Vec2) { m_data.asVec2 = data; }
		Property(const char* name, vec3* data) : m_name(name), m_type(PropertyType_Vec3) { m_data.asVec3 = data; }
		Property(const char* name, vec4* data) : m_name(name), m_type(PropertyType_Vec4) { m_data.asVec4 = data; }		

		const char* m_name;
		PropertyType m_type;
		union
		{
			bool* asBool;
			float* asFloat;
			int* asInt;
			vec2* asVec2;
			vec3* asVec3;
			vec4* asVec4;
		} m_data;
	};

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
				vec2 uv = (vec2(pixelPos) + vec2(0.5f)) / vec2(m_outputSize);
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

	virtual void getProperties(std::vector<Property>& outProperties)
	{
		outProperties.push_back(Property("Enabled", &m_enabled));
	}

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

        // Compute max irradiance sample
        m_irradianceMax = FLT_MIN;
        m_irradianceImage.forPixels2D([&](const vec4& v, ivec2 pixelPos)
        {
            m_irradianceMax = std::max((0.299f * v.r + 0.587f * v.g + 0.114f * v.b), m_irradianceMax);
        });

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
    float m_irradianceMax = 0.0f;
};

typedef std::vector<std::unique_ptr<Experiment>> ExperimentList;

void addAllExperiments(ExperimentList& experiments);
void resetAllExperiments(ExperimentList& experiments);

} // namespace Probulator