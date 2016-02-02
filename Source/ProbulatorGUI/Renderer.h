#pragma once

#include "Common.h"

#include <Probulator/Math.h>

#include <memory>
#include <GL/gl3w.h>

namespace Probulator
{
	class Image;
}

enum VertexAttribute
{
	VertexAttribute_Invalid,
	VertexAttribute_Position,
	VertexAttribute_Normal,
	VertexAttribute_TexCoord0,
	VertexAttribute_TexCoord1,
	VertexAttribute_TexCoord2,
	VertexAttribute_TexCoord3,
};

const char* toString(VertexAttribute attribute);

struct VertexElement
{
	VertexAttribute attribute = VertexAttribute_Invalid;
	u32 dataType = GL_FLOAT;
	GLboolean normalized = GL_FALSE;
	u32 componentCount = 1;
	u32 offset = 0;
};

struct VertexFormat
{
	enum { MaxElements = 8 };

	u32 elementCount = 0;
	VertexElement elements[MaxElements];

	VertexFormat& add(VertexAttribute attribute, u32 dataType, GLboolean normalized, u32 componentCount, u32 offset)
	{
		assert(elementCount < MaxElements);
		VertexElement& element = elements[elementCount];
		element.attribute = attribute;
		element.dataType = dataType;
		element.normalized = normalized;
		element.componentCount = componentCount;
		element.offset = offset;
		elementCount++;
		return *this;
	}
};

struct TextureFilter
{
	u32 wrapU = GL_REPEAT;
	u32 wrapV = GL_REPEAT;
	u32 filterMin = GL_NEAREST;
	u32 filterMag = GL_NEAREST;
};

inline TextureFilter makeTextureFilter(u32 wrap, u32 filter)
{
	TextureFilter result;
	result.wrapU = wrap;
	result.wrapV = wrap;
	result.filterMin = filter;
	result.filterMag = filter;
	return result;
}

class Texture : NonCopyable
{
public:

	~Texture()
	{
		glDeleteTextures(1, &m_native);
	}

	u32 m_native = 0;
	u32 m_type = 0;
};

class Shader : NonCopyable
{
public:

	~Shader()
	{
		glDeleteShader(m_native);
	}

	u32 m_native = 0;
	u32 m_type = 0;
};

class ShaderProgram : NonCopyable
{
public:

	enum { MaxTextures = 8 };

	~ShaderProgram()
	{
		glDeleteProgram(m_native);
		glDeleteVertexArrays(1, &m_vertexArray);
	}

	u32 m_native = 0;
	u32 m_vertexArray = 0;
	VertexFormat m_vertexFormat;
	GLint m_vertexAttributeLocations[VertexFormat::MaxElements];
	GLint m_textureLocations[MaxTextures];
};

typedef std::shared_ptr<Texture> TexturePtr;
typedef std::shared_ptr<Shader> ShaderPtr;
typedef std::shared_ptr<ShaderProgram> ShaderProgramPtr;

TexturePtr createTextureFromImage(
	const Image& image,
	const TextureFilter& filter = TextureFilter(),
	bool verticalFlip = false);

ShaderPtr createShaderFromSource(u32 type, const char* source);

ShaderProgramPtr createShaderProgram(
	const Shader& vertexShader,
	const Shader& pixelShader,
	const VertexFormat& vertexFormat);

void setTexture(const ShaderProgram& shaderProgram, u32 slotIndex, const Texture& texture);
void setVertexBuffer(const ShaderProgram& shaderProgram, u32 vertexBuffer, u32 vertexStride);

void setUniformByName(const ShaderProgram& shaderProgram, const char* name, float value);
void setUniformByName(const ShaderProgram& shaderProgram, const char* name, const vec2& value);
void setUniformByName(const ShaderProgram& shaderProgram, const char* name, const mat4& value);
