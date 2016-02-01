#pragma once

#include "Common.h"
#include "Renderer.h"
#include "Shaders.h"

#include <Probulator/Math.h>

class Blitter : NonCopyable
{
public:

	Blitter();
	~Blitter();

	void drawTexture(const ShaderProgram& shaderProgram, const Texture& texture);
	void drawTexture(const ShaderProgram& shaderProgram, const CommonShaderUniforms& shaderUniforms, const Texture& texture);

private:

	struct Vertex
	{
		float x, y;
	};

	u32 m_vertexBuffer = 0;
};
