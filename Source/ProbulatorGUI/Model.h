#include "Common.h"
#include "Renderer.h"
#include "Shaders.h"

#include <Probulator/Math.h>

class Model : NonCopyable
{
public:

	struct ProceduralSphere
	{
		u64 numUSlices = 256;
		u64 numVSlices = 256;
	};


	Model(const char* objFilename);
	Model(const ProceduralSphere& sphere);
	~Model();

	struct Vertex
	{
		vec3 position;
		vec3 normal;
		vec2 texCoord;
	};

	bool readObj(const char* objFilename, bool forceGenerateNormals);
    void generateSphere(u64 numUSlices = 256, u64 numVSlices = 192);

	bool m_valid = false;
	
	u32 m_vertexBuffer = 0;
	u32 m_indexBuffer = 0;
	u32 m_indexCount = 0;

	vec3 m_boundsMin = vec3(0.0f);
	vec3 m_boundsMax = vec3(0.0f);
	vec3 m_dimensions = vec3(0.0f);
	vec3 m_center = vec3(0.0f);

	void draw(
		const ShaderProgram& shaderProgram,
		const CommonShaderUniforms& shaderUniforms,
		const Texture& irradianceTexture,
		const mat4& worldMatrix);
};
