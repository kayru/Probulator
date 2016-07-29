#include "Common.h"
#include "Renderer.h"
#include "Shaders.h"

#include <Probulator/Math.h>
#include <vector>

class Model : NonCopyable
{
public:

	struct ProceduralSphere
	{
		u64 numUSlices = 256;
		u64 numVSlices = 256;
	};

	struct ProceduralPlane
	{
		vec2 dimensions = vec2(1.0f);
	};


	Model(const char* objFilename);
	Model(const ProceduralSphere& sphere);
	Model(const ProceduralPlane& plane);
	~Model();

	struct Vertex
	{
		vec3 position;
		vec3 normal;
		vec2 texCoord;
	};

	bool readObj(const char* objFilename, bool forceGenerateNormals);
    void generateSphere(u64 numUSlices = 256, u64 numVSlices = 192);
	void generatePlane(const ProceduralPlane& plane);

	void createBuffers(const Vertex* vertices, u32 vertexCount, const u32* indices, u32 indexCount);
	void createBuffers(const std::vector<Vertex>& vertices, const std::vector<u32>& indices);

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
