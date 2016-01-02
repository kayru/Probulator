#include "Blitter.h"

static const char* g_vertexShaderSource =
R"(#version 150
in vec2 Position;
in vec2 TexCoord0;
void main()
{
	gl_Position = vec4(Position.x, Position.y, 0.0, 1.0);
}
)";

static const char* g_pixelShaderSourceTexture2D =
R"(#version 150
out vec4 Target;
void main()
{
	Target = vec4(1.0, 0.5, 0.0, 1.0);
}
)";


Blitter::Blitter()
{
	VertexDeclaration vertexDeclaration;
	vertexDeclaration.add(VertexSemantic_Position, GL_FLOAT, 2, 0);

	m_vertexShader = createShaderFromSource(GL_VERTEX_SHADER, g_vertexShaderSource);
	m_pixelShaderTexture2D = createShaderFromSource(GL_FRAGMENT_SHADER, g_pixelShaderSourceTexture2D);

	m_programTexture2D = createShaderProgram(
		*m_vertexShader,
		*m_pixelShaderTexture2D,
		vertexDeclaration);

	float vertices[6] =
	{
		-1.0f, -1.0f,
		-1.0f,  3.0f,
		 3.0f, -1.0f,
	};
	glGenBuffers(1, &m_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

Blitter::~Blitter()
{
	glDeleteBuffers(1, &m_vertexBuffer);
}

void Blitter::drawTexture2D(Texture* texture)
{

}
