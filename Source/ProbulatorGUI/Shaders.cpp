#include "Shaders.h"
#include "Model.h"

#include <sstream>
#include <stdio.h>

static std::string pathFromFilename(const std::string& filename)
{
	size_t pos = filename.find_last_of("/\\");
	if (pos != std::string::npos)
	{
		return filename.substr(0, pos + 1);
	}
	else
	{
		return std::string();
	}
}

static std::string stringFromFile(const std::string& filename)
{
	std::string res;

	FILE* f = fopen(filename.c_str(), "rb");

	if (!f)
	{
		return res;
	}

	fseek(f, 0, SEEK_END);
	u64 fileSize = ftell(f);

	fseek(f, 0, SEEK_SET);

	res.resize(fileSize);

	fread(&res[0], fileSize, 1, f);
	fclose(f);

	return res;
}

static std::string preprocessShaderFromFile(const std::string& filename)
{
	std::string shaderSource = stringFromFile(filename);
	std::string shaderPath = pathFromFilename(filename);

	std::stringstream result;
	std::istringstream stream(shaderSource);

	std::string line;

	u32 currentLine = 1;
	while (std::getline(stream, line))
	{
		if (line.find("#include") == 0)
		{
			size_t p0 = line.find_first_of("\"<");
			size_t p1 = line.find_last_of("\">");
			if (p0 != p1 && p0 != std::string::npos && p1 != std::string::npos)
			{
				std::string includeFilename = line.substr(p0 + 1, p1 - p0 - 1);
				result << preprocessShaderFromFile(shaderPath + includeFilename) << "\n";
				++currentLine;
				result << "#line " << currentLine << "\n";
			}
		}
		else
		{
			result << line << "\n";
		}

		++currentLine;
	}

	return result.str();
}

ShaderPtr createShaderFromFile(u32 type, const char* filename)
{
	std::string shaderSource = preprocessShaderFromFile(filename);
	return createShaderFromSource(type, shaderSource.c_str());
}

void setCommonUniforms(const ShaderProgram& shaderProgram, const CommonShaderUniforms& uniforms)
{
	setUniformByName(shaderProgram, "uElapsedTime", uniforms.elapsedTime);
	setUniformByName(shaderProgram, "uExposure", uniforms.exposure);
	setUniformByName(shaderProgram, "uProjMatrix", uniforms.projMatrix);
	setUniformByName(shaderProgram, "uResolution", uniforms.resolution);
	setUniformByName(shaderProgram, "uViewMatrix", uniforms.viewMatrix);
	setUniformByName(shaderProgram, "uViewProjMatrix", uniforms.viewProjMatrix);
}

CommonShaderPrograms::CommonShaderPrograms()
{
	// Blitter

	ShaderPtr vsBlitDefault = createShaderFromFile(GL_VERTEX_SHADER, "Data/Shaders/Blitter.vert");
	ShaderPtr psBlitTexture2D = createShaderFromFile(GL_FRAGMENT_SHADER, "Data/Shaders/BlitterTexture2D.frag");
	ShaderPtr psBlitLatLongEnvmap = createShaderFromFile(GL_FRAGMENT_SHADER, "Data/Shaders/BlitterLatLongEnvmap.frag");

	VertexFormat vfBlit;
	vfBlit.add(VertexAttribute_Position, GL_FLOAT, GL_FALSE, 2, 0);

	blitTexture2D = createShaderProgram(*vsBlitDefault, *psBlitTexture2D, vfBlit);
	blitLatLongEnvmap = createShaderProgram(*vsBlitDefault, *psBlitLatLongEnvmap, vfBlit);

	// Model

	VertexFormat vfModel = VertexFormat()
		.add(VertexAttribute_Position, GL_FLOAT, false, 3, offsetof(Model::Vertex, position))
		.add(VertexAttribute_Normal, GL_FLOAT, false, 3, offsetof(Model::Vertex, normal))
		.add(VertexAttribute_TexCoord0, GL_FLOAT, false, 2, offsetof(Model::Vertex, texCoord));

	auto vsModelDefault = createShaderFromFile(GL_VERTEX_SHADER, "Data/Shaders/Model.vert");
	auto vsModelBasisVisualizer = createShaderFromFile(GL_VERTEX_SHADER, "Data/Shaders/BasisVisualizer.vert");
	auto psModelIrradianceEnvmap = createShaderFromFile(GL_FRAGMENT_SHADER, "Data/Shaders/ModelIrradianceEnvmap.frag");
	auto psModelIrradianceLightmap = createShaderFromFile(GL_FRAGMENT_SHADER, "Data/Shaders/ModelIrradianceLightmap.frag");

	modelIrradianceLightmap = createShaderProgram(*vsModelDefault, *psModelIrradianceLightmap, vfModel);
	modelIrradianceEnvmap = createShaderProgram(*vsModelDefault, *psModelIrradianceEnvmap, vfModel);
	modelBasisVisualizer = createShaderProgram(*vsModelBasisVisualizer, *psModelIrradianceEnvmap, vfModel);
}
