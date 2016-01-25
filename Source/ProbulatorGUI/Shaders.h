#pragma once

#include "Common.h"
#include "Renderer.h"

#include <string>

struct CommonShaderUniforms
{
	float exposure = 1.0f;
	mat4 viewMatrix = mat4(1.0f);
	mat4 projMatrix = mat4(1.0f);
	mat4 viewProjMatrix = mat4(1.0f);
};

// Load shader from file with minimal preprocess step that handles include directives.
// Included files must be in the same directory as source file.
std::string preprocessShaderFromFile(const std::string& filename);
ShaderPtr createShaderFromFile(u32 type, const char* filename);
