#pragma once

#include "Common.h"
#include "Renderer.h"

#include <string>

struct CommonShaderUniforms
{
	float elapsedTime = 0.0f;
	float exposure = 1.0f;
	vec2 resolution = vec2(1.0f);
	mat4 viewMatrix = mat4(1.0f);
	mat4 projMatrix = mat4(1.0f);
	mat4 viewProjMatrix = mat4(1.0f);
};

class CommonShaderPrograms : NonCopyable
{
public:
	CommonShaderPrograms();

	ShaderProgramPtr blitTexture2D;
	ShaderProgramPtr blitLatLongEnvmap;

	ShaderProgramPtr modelIrradianceLightmap;
	ShaderProgramPtr modelIrradianceEnvmap;
	ShaderProgramPtr modelBasisVisualizer;
};

// Load shader from file with minimal preprocess step that handles include directives.
// Included files must be in the same directory as source file.
ShaderPtr createShaderFromFile(u32 type, const char* filename);

void setCommonUniforms(const ShaderProgram& shaderProgram, const CommonShaderUniforms& uniforms);
