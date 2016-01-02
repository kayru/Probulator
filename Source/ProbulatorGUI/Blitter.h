#pragma once

#include "Common.h"
#include "Renderer.h"

class Blitter : NonCopyable
{
public:

	Blitter();
	~Blitter();

	void drawTexture2D(Texture* texture);

private:

	struct Vertex
	{
		float position[2];
	};

	u32 m_vertexBuffer = 0;

	ShaderPtr m_vertexShader;
	ShaderPtr m_pixelShaderTexture2D;
	ShaderProgramPtr m_programTexture2D;
};
