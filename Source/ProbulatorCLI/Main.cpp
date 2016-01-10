#include <Probulator/Experiments.h>

#include <stdio.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <string.h>

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

using namespace Probulator;

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
    addAllExperiments(experiments);

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