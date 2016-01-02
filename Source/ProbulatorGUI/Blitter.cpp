#include "Blitter.h"

static const char* g_vertexShaderSource =
R"(#version 150
in vec2 Position;
in vec2 TexCoord0;
out vec2 vTexCoord0;
void main()
{
	gl_Position = vec4(Position.x, Position.y, 1.0, 1.0);
	vTexCoord0 = Position * 0.5 + 0.5;
}
)";

static const char* g_pixelShaderSourceTexture2D =
R"(#version 150
uniform sampler2D Texture0;
out vec4 Target;
in vec2 vTexCoord0;
void main()
{
	Target = texture(Texture0, vTexCoord0);
}
)";

Blitter::Blitter()
{
	VertexDeclaration vertexDeclaration;
	vertexDeclaration.add(VertexSemantic_Position, GL_FLOAT, GL_FALSE, 2, 0);

	m_vertexShader = createShaderFromSource(GL_VERTEX_SHADER, g_vertexShaderSource);
	m_pixelShaderTexture2D = createShaderFromSource(GL_FRAGMENT_SHADER, g_pixelShaderSourceTexture2D);

	m_programTexture2D = createShaderProgram(
		*m_vertexShader,
		*m_pixelShaderTexture2D,
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

	glDrawArrays(GL_TRIANGLES, 0, 3);
}
