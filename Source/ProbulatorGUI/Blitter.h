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

	void drawTexture2D(const Texture& texture);
	void drawLatLongEnvmap(const Texture& texture, const CommonShaderUniforms& shaderUniforms);

private:

	struct Vertex
	{
		float x, y;
	};

	u32 m_vertexBuffer = 0;

	ShaderProgramPtr m_programTexture2D;
	ShaderProgramPtr m_programLatLongEnvmap;
};
