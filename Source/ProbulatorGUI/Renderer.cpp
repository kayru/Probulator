#include "Renderer.h"

#include <Probulator/Image.h>
#include <string>

TexturePtr createTextureFromImage(const Image& image, const TextureFilter& filter)
{
	assert(image.getPixelCount() != 0);

	TexturePtr result = std::make_shared<Texture>();

	const GLenum type = GL_TEXTURE_2D;
	result->m_type = type;

	glGenTextures(1, &result->m_native);
	glBindTexture(result->m_type, result->m_native);

	glTexParameteri(type, GL_TEXTURE_WRAP_S, filter.wrapU);
	glTexParameteri(type, GL_TEXTURE_WRAP_T, filter.wrapV);
	glTexParameteri(type, GL_TEXTURE_MIN_FILTER, filter.filterMin);
	glTexParameteri(type, GL_TEXTURE_MAG_FILTER, filter.filterMag);

	const GLsizei w = image.getWidth();
	const GLsizei h = image.getHeight();
	glTexImage2D(type, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_FLOAT, image.data());

	return result;
}

const char* toString(VertexSemantic semantic)
{
	switch(semantic)
	{
		default:
		case VertexSemantic_Invalid:
			return "Invalid";
		case VertexSemantic_Position:
			return "Position";
		case VertexSemantic_TexCoord0:
			return "TexCoord0";
		case VertexSemantic_TexCoord1:
			return "TexCoord1";
		case VertexSemantic_TexCoord2:
			return "TexCoord2";
		case VertexSemantic_TexCoord3:
			return "TexCoord3";
	}
}

ShaderPtr createShaderFromSource(u32 type, const char* source)
{
	ShaderPtr result = std::make_shared<Shader>();

	result->m_type = type;
	result->m_native = glCreateShader(type);
	glShaderSource(result->m_native, 1, &source, nullptr);
	glCompileShader(result->m_native);

	GLint compileStatus = 0;
	glGetShaderiv(result->m_native, GL_COMPILE_STATUS, &compileStatus);
	if (!compileStatus)
	{
		GLint infoLogLength = 0;
		glGetShaderiv(result->m_native, GL_INFO_LOG_LENGTH, &infoLogLength);
		std::string shaderInfoLog;
		shaderInfoLog.resize(infoLogLength);
		glGetShaderInfoLog(result->m_native, infoLogLength, nullptr, &shaderInfoLog[0]);
		printf("ERROR: Could not compile shader.\n%s\n", shaderInfoLog.c_str());
		assert(compileStatus && "Could not compile shader");
	}
	return result;
}


ShaderProgramPtr createShaderProgram(
	const Shader& vertexShader,
	const Shader& pixelShader,
	const VertexDeclaration& vertexDeclaration)
{
	assert(vertexShader.m_native);
	assert(pixelShader.m_native);

	ShaderProgramPtr result = std::make_shared<ShaderProgram>();

	result->m_native = glCreateProgram();
	result->m_vertexDeclaration = vertexDeclaration;

	glAttachShader(result->m_native, vertexShader.m_native);
	glAttachShader(result->m_native, pixelShader.m_native);
	glLinkProgram(result->m_native);

	GLint linkStatus = 0;
	glGetProgramiv(result->m_native, GL_LINK_STATUS, &linkStatus);

	assert(linkStatus && "Could not link shader program"); // TODO: log linking errors

	glGenVertexArrays(1, &result->m_vertexArray);
	glBindVertexArray(result->m_vertexArray);

	for (u32 i=0; i<vertexDeclaration.elementCount; ++i)
	{
		const VertexElement& element = vertexDeclaration.elements[i];
		const char* semanticName = toString(element.semantic);
		GLint location = glGetAttribLocation(result->m_native, semanticName);
		if (location != -1)
		{
			glEnableVertexAttribArray(location);
		}
		result->m_vertexAttributeLocations[i] = location;
	}

	for (u32 i=vertexDeclaration.elementCount; i<VertexDeclaration::MaxElements; ++i)
	{
		result->m_vertexAttributeLocations[i] = -1;
	}

	return result;
}
