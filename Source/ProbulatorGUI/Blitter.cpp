#include "Blitter.h"

static const char* g_vertexShaderSource =
R"(#version 150
in vec2 Position;
in vec2 TexCoord0;
out vec2 vTexCoord0;
out vec4 vPosition;
void main()
{
	vPosition = vec4(Position.x, Position.y, 1.0, 1.0);
	vTexCoord0 = Position * 0.5 + 0.5;
	vTexCoord0.y = 1.0 - vTexCoord0.y;
	gl_Position = vPosition;
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

static const char* g_pixelShaderSourceLatLongEnvmap =
R"(#version 150
uniform mat4 uViewMatrix;
uniform mat4 uProjMatrix;
uniform sampler2D Texture0;
out vec4 Target;
in vec4 vPosition;
#define PI 3.14159265358979323846
vec2 cartesianToLatLongTexcoord(vec3 p)
{
	float u = (1.0 + atan(p.x, -p.z) / PI);
	float v = acos(p.y) / PI;
	return vec2(u * 0.5, v);
}
void main()
{
	vec3 view;
	view.x = vPosition.x / uProjMatrix[0][0];
	view.y = vPosition.y / uProjMatrix[1][1];
	view.z = -1.0;
	view = normalize(view * mat3(uViewMatrix));
	vec2 texCoord = cartesianToLatLongTexcoord(view);
	Target = texture(Texture0, texCoord);
}
)";

Blitter::Blitter()
{
	VertexDeclaration vertexDeclaration;
	vertexDeclaration.add(VertexAttribute_Position, GL_FLOAT, GL_FALSE, 2, 0);

	ShaderPtr vertexShader = createShaderFromSource(GL_VERTEX_SHADER, g_vertexShaderSource);
	ShaderPtr pixelShaderTexture2D = createShaderFromSource(GL_FRAGMENT_SHADER, g_pixelShaderSourceTexture2D);
	ShaderPtr pixelShaderLatLongEnvmap = createShaderFromSource(GL_FRAGMENT_SHADER, g_pixelShaderSourceLatLongEnvmap);

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
