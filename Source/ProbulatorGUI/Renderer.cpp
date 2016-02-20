#include "Renderer.h"

#include <Probulator/Image.h>
#include <string>

#include <glm/gtc/type_ptr.hpp>

TexturePtr createTextureFromImage(
	const Image& image,
	const TextureFilter& filter,
	bool verticalFlip)
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

	if (verticalFlip)
	{
		Image flippedImage(w, h);
		for (int y = 0; y < h; ++y)
		{
			vec4* rowPixelsOut = flippedImage.getPixels() + w*y;
			const vec4* rowPixelsIn = image.getPixels() + w*(h - 1 - y);
			memcpy(rowPixelsOut, rowPixelsIn, sizeof(vec4) * w);
		}

		glTexImage2D(type, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, flippedImage.data());
	}
	else
	{
		glTexImage2D(type, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, image.data());
	}


	return result;
}

const char* toString(VertexAttribute attribute)
{
	switch (attribute)
	{
	default:
	case VertexAttribute_Invalid:
		return "Invalid";
	case VertexAttribute_Position:
		return "Position";
	case VertexAttribute_Normal:
		return "Normal";
	case VertexAttribute_TexCoord0:
		return "TexCoord0";
	case VertexAttribute_TexCoord1:
		return "TexCoord1";
	case VertexAttribute_TexCoord2:
		return "TexCoord2";
	case VertexAttribute_TexCoord3:
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

		throw std::runtime_error("Failed to compile shader.");
	}
	return result;
}


ShaderProgramPtr createShaderProgram(
	const Shader& vertexShader,
	const Shader& pixelShader,
	const VertexFormat& vertexFormat)
{
	assert(vertexShader.m_native);
	assert(pixelShader.m_native);

	ShaderProgramPtr result = std::make_shared<ShaderProgram>();

	result->m_native = glCreateProgram();
	result->m_vertexFormat = vertexFormat;

	glAttachShader(result->m_native, vertexShader.m_native);
	glAttachShader(result->m_native, pixelShader.m_native);
	glLinkProgram(result->m_native);

	GLint linkStatus = 0;
	glGetProgramiv(result->m_native, GL_LINK_STATUS, &linkStatus);

    if (!linkStatus)
    {
        GLchar errorLog[1024] = { 0 };
        glGetProgramInfoLog(result->m_native, 1024, NULL, errorLog);
        
        printf((std::string() + "error linking program: " + errorLog).c_str());
		throw std::runtime_error("Failed to link shader program.");
    }

	glGenVertexArrays(1, &result->m_vertexArray);
	glBindVertexArray(result->m_vertexArray);

	for (u32 i = 0; i < vertexFormat.elementCount; ++i)
	{
		const VertexElement& element = vertexFormat.elements[i];
		const char* attributeName = toString(element.attribute);
		GLint location = glGetAttribLocation(result->m_native, attributeName);
		if (location != -1)
		{
			glEnableVertexAttribArray(location);
		}
		result->m_vertexAttributeLocations[i] = location;
	}

	for (u32 i = vertexFormat.elementCount; i < VertexFormat::MaxElements; ++i)
	{
		result->m_vertexAttributeLocations[i] = -1;
	}

	for (u32 i = 0; i < ShaderProgram::MaxTextures; ++i)
	{
		char name[] = "Texture#";
		name[7] = '0' + i;
		result->m_textureLocations[i] = glGetUniformLocation(result->m_native, name);
	}

	return result;
}

void setVertexBuffer(const ShaderProgram& shaderProgram, u32 vertexBuffer, u32 vertexStride)
{
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBindVertexArray(shaderProgram.m_vertexArray);
	for (u32 i = 0; i < shaderProgram.m_vertexFormat.elementCount; ++i)
	{
		const VertexElement& element = shaderProgram.m_vertexFormat.elements[i];
		glVertexAttribPointer(
			shaderProgram.m_vertexAttributeLocations[i],
			element.componentCount,
			element.dataType,
			element.normalized,
			vertexStride,
			((char*)(nullptr)) + element.offset);
	}
}

void setTexture(const ShaderProgram& shaderProgram, u32 slotIndex, const Texture& texture)
{
	GLint location = shaderProgram.m_textureLocations[slotIndex];

	if (location == -1)
		return;

	glActiveTexture(GL_TEXTURE0 + slotIndex);
	glBindTexture(texture.m_type, texture.m_native);
	glUniform1i(location, slotIndex);
}

void setUniformByName(const ShaderProgram& shaderProgram, const char* name, float value)
{
	GLint location = glGetUniformLocation(shaderProgram.m_native, name); // TODO: cache uniform bindings

	if (location == -1)
		return;

	glUniform1fv(location, 1, &value);
}

void setUniformByName(const ShaderProgram& shaderProgram, const char* name, const mat4& value)
{
	GLint location = glGetUniformLocation(shaderProgram.m_native, name); // TODO: cache uniform bindings

	if (location == -1)
		return;

	glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(value));
}

void setUniformByName(const ShaderProgram& shaderProgram, const char* name, const vec2& value)
{
	GLint location = glGetUniformLocation(shaderProgram.m_native, name); // TODO: cache uniform bindings

	if (location == -1)
		return;

	glUniform2fv(location, 1, glm::value_ptr(value));
}

