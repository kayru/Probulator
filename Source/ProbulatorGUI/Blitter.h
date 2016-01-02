#pragma once

#include "Common.h"
#include "Renderer.h"

class Blitter : NonCopyable
{
public:

	Blitter();
	~Blitter();

	void drawTexture2D(const Texture& texture);

private:

	struct Vertex
	{
		float x, y;
	};

	u32 m_vertexBuffer = 0;

	ShaderPtr m_vertexShader;
	ShaderPtr m_pixelShaderTexture2D;
	ShaderProgramPtr m_programTexture2D;
};
