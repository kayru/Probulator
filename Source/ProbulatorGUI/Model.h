#include "Common.h"
#include "Renderer.h"

#include <Probulator/Math.h>

class Model : NonCopyable
{
public:

	Model(const char* objFilename);
	~Model();

	struct Vertex
	{
		vec3 position;
		vec3 normal;
		vec2 texCoord;
	};

	bool readObj(const char* objFilename, bool forceGenerateNormals);

	bool m_valid = false;
	
	u32 m_vertexBuffer = 0;
	u32 m_indexBuffer = 0;
	u32 m_indexCount = 0;

	vec3 m_boundsMin = vec3(0.0f);
	vec3 m_boundsMax = vec3(0.0f);
	vec3 m_dimensions = vec3(0.0f);
	vec3 m_center = vec3(0.0f);

	VertexDeclaration m_vertexDeclaration = VertexDeclaration()
		.add(VertexAttribute_Position, GL_FLOAT, false, 3, offsetof(Vertex, position))
		.add(VertexAttribute_Normal, GL_FLOAT, false, 3, offsetof(Vertex, normal))
		.add(VertexAttribute_TexCoord0, GL_FLOAT, false, 2, offsetof(Vertex, texCoord));

	void draw(
		const Texture& irradianceTexture,
		const mat4& worldMatrix, 
		const mat4& viewProjMatrix);

	ShaderProgramPtr m_shaderProgram;
};
