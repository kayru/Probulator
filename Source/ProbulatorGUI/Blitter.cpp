#include "Blitter.h"
#include "Shaders.h"

Blitter::Blitter()
{
	VertexDeclaration vertexDeclaration;
	vertexDeclaration.add(VertexAttribute_Position, GL_FLOAT, GL_FALSE, 2, 0);

	ShaderPtr vertexShader = createShaderFromFile(GL_VERTEX_SHADER, "Data/Shaders/Blitter.vert");
	ShaderPtr pixelShaderTexture2D = createShaderFromFile(GL_FRAGMENT_SHADER, "Data/Shaders/BlitterTexture2D.frag");
	ShaderPtr pixelShaderLatLongEnvmap = createShaderFromFile(GL_FRAGMENT_SHADER, "Data/Shaders/BlitterLatLongEnvmap.frag");

	m_programTexture2D = createShaderProgram(
		*vertexShader,
		*pixelShaderTexture2D,
		vertexDeclaration);

	m_programLatLongEnvmap = createShaderProgram(
		*vertexShader,
		*pixelShaderLatLongEnvmap,
		vertexDeclaration);

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

void Blitter::drawTexture2D(const Texture& texture)
{
	glUseProgram(m_programTexture2D->m_native);
	
	setTexture(*m_programTexture2D, 0, texture);
	setVertexBuffer(*m_programTexture2D, m_vertexBuffer, sizeof(Vertex));

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);

	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Blitter::drawLatLongEnvmap(const Texture& texture, const mat4& viewMatrix, const mat4& projectionMatrix)
{
	glUseProgram(m_programLatLongEnvmap->m_native);

	setTexture(*m_programLatLongEnvmap, 0, texture);
	setVertexBuffer(*m_programLatLongEnvmap, m_vertexBuffer, sizeof(Vertex));
	setUniformByName(*m_programLatLongEnvmap, "uViewMatrix", viewMatrix);
	setUniformByName(*m_programLatLongEnvmap, "uProjMatrix", projectionMatrix);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);

	glDrawArrays(GL_TRIANGLES, 0, 3);
}
