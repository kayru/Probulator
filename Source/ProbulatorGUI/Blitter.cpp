#include "Blitter.h"

Blitter::Blitter()
{
	const Vertex vertices[3] =
	{
		{-1.0f, -1.0f},
		{-1.0f,  3.0f},
		{ 3.0f, -1.0f},
	};

	glGenBuffers(1, &m_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

Blitter::~Blitter()
{
	glDeleteBuffers(1, &m_vertexBuffer);
}

void Blitter::drawTexture(const ShaderProgram& shaderProgram, const Texture& texture)
{
	glUseProgram(shaderProgram.m_native);
	
	setTexture(shaderProgram, 0, texture);
	setVertexBuffer(shaderProgram, m_vertexBuffer, sizeof(Vertex));

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);

	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Blitter::drawTexture(const ShaderProgram& shaderProgram, const CommonShaderUniforms& shaderUniforms, const Texture& texture)
{
	glUseProgram(shaderProgram.m_native);

	setTexture(shaderProgram, 0, texture);
	setVertexBuffer(shaderProgram, m_vertexBuffer, sizeof(Vertex));

	setCommonUniforms(shaderProgram, shaderUniforms);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);

	glDrawArrays(GL_TRIANGLES, 0, 3);
}
