#pragma once

#include "Common.h"

#include <GL/gl3w.h>
#include <memory>

namespace Probulator
{
	class Image;
}

enum VertexSemantic
{
	VertexSemantic_Invalid,
	VertexSemantic_Position,
	VertexSemantic_TexCoord0,
	VertexSemantic_TexCoord1,
	VertexSemantic_TexCoord2,
	VertexSemantic_TexCoord3,
};

const char* toString(VertexSemantic semantic);

struct VertexElement
{
	VertexSemantic semantic = VertexSemantic_Invalid;
	u32 dataType = GL_FLOAT;
	u32 componentCount = 1;
	u32 offset = 0;
};

struct VertexDeclaration
{
	enum { MaxElements = 8 };

	u32 elementCount = 0;
	VertexElement elements[MaxElements];

	VertexDeclaration& add(VertexSemantic semantic, u32 dataType, u32 componentCount, u32 offset)
	{
		assert(elementCount < MaxElements);
		VertexElement& element = elements[elementCount];
		element.semantic = semantic;
		element.dataType = dataType;
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

	~ShaderProgram()
	{
		glDeleteProgram(m_native);
		glDeleteVertexArrays(1, &m_vertexArray);
	}

	u32 m_native = 0;
	u32 m_vertexArray = 0;
	VertexDeclaration m_vertexDeclaration;
	GLint m_vertexAttributeLocations[VertexDeclaration::MaxElements];
};

typedef std::shared_ptr<Texture> TexturePtr;
typedef std::shared_ptr<Shader> ShaderPtr;
typedef std::shared_ptr<ShaderProgram> ShaderProgramPtr;

TexturePtr createTextureFromImage(
	const Image& image,
	const TextureFilter& filter = TextureFilter());

ShaderPtr createShaderFromSource(u32 type, const char* source);

ShaderProgramPtr createShaderProgram(
	const Shader& vertexShader,
	const Shader& pixelShader,
	const VertexDeclaration& vertexDeclaration);
