#include "Model.h"
#include "Shaders.h"

#include <tiny_obj_loader.h>
#include <vector>

Model::Model(const char* objFilename)
{
	auto vertexShader = createShaderFromFile(GL_VERTEX_SHADER, "Data/Shaders/Model.vert");
	auto pixelShader = createShaderFromFile(GL_FRAGMENT_SHADER, "Data/Shaders/Model.frag");

	m_shaderProgram = createShaderProgram(
		*vertexShader,
		*pixelShader,
		m_vertexDeclaration);

	const bool forceGenerateNormals = false;
	readObj(objFilename, forceGenerateNormals);
}

Model::~Model()
{
	glDeleteBuffers(1, &m_vertexBuffer);
	glDeleteBuffers(1, &m_indexBuffer);
}

bool Model::readObj(const char* objFilename, bool forceGenerateNormals)
{
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string errors;

	m_valid = tinyobj::LoadObj(shapes, materials, errors, objFilename);
	if (!m_valid)
	{
		printf("ERROR: Could not load model from '%s'\n%s\n", objFilename, errors.c_str());
		return m_valid;
	}

	std::vector<Vertex> vertices;
	std::vector<u32> indices;

	m_boundsMin = vec3(FLT_MAX);
	m_boundsMax = vec3(-FLT_MAX);

	for (const auto& shape : shapes)
	{
		u32 firstVertex = (u32)vertices.size();
		const auto& mesh = shape.mesh;

		const u32 vertexCount = (u32)mesh.positions.size() / 3;

		const bool haveNormals = mesh.positions.size() == mesh.normals.size();
		const bool haveTexCoords = mesh.positions.size() == mesh.texcoords.size();

		for (u32 i = 0; i < vertexCount; ++i)
		{
			Vertex v = {};

			v.position.x = mesh.positions[i * 3 + 0];
			v.position.y = mesh.positions[i * 3 + 1];
			v.position.z = mesh.positions[i * 3 + 2];

			if (haveTexCoords)
			{
				v.texCoord.x = mesh.texcoords[i * 2 + 0];
				v.texCoord.y = mesh.texcoords[i * 2 + 1];
			}

			if (haveNormals)
			{
				v.normal.x = mesh.normals[i * 3 + 0];
				v.normal.y = mesh.normals[i * 3 + 1];
				v.normal.z = mesh.normals[i * 3 + 2];
			}

			vertices.push_back(v);

			m_boundsMin = min(m_boundsMin, v.position);
			m_boundsMax = max(m_boundsMax, v.position);
		}

		if (!haveNormals || forceGenerateNormals)
		{
			const u32 triangleCount = (u32)mesh.indices.size() / 3;
			for (u32 i = 0; i < triangleCount; ++i)
			{
				u32 idxA = firstVertex + mesh.indices[i * 3 + 0];
				u32 idxB = firstVertex + mesh.indices[i * 3 + 1];
				u32 idxC = firstVertex + mesh.indices[i * 3 + 2];

				vec3 a = vertices[idxA].position;
				vec3 b = vertices[idxB].position;
				vec3 c = vertices[idxC].position;

				vec3 normal = cross(b - a, c - b);

				normal = normalize(normal);

				vertices[idxA].normal += normal;
				vertices[idxB].normal += normal;
				vertices[idxC].normal += normal;
			}

			for (u32 i = firstVertex; i < (u32)vertices.size(); ++i)
			{
				vertices[i].normal = normalize(vertices[i].normal);
			}
		}

		for (u32 index : mesh.indices)
		{
			indices.push_back(index + firstVertex);
		}

		m_indexCount = (u32)indices.size();
	}

	m_dimensions = m_boundsMax - m_boundsMin;
	m_center = (m_boundsMax + m_boundsMin) / 2.0f;

	glGenBuffers(1, &m_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

	glGenBuffers(1, &m_indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(u32), indices.data(), GL_STATIC_DRAW);

	return m_valid;
}

void Model::draw(
	const Texture& irradianceTexture,
	const CommonShaderUniforms& shaderUniforms,
	const mat4& worldMatrix)
{
	if (!m_valid)
		return;

	glUseProgram(m_shaderProgram->m_native);

	setTexture(*m_shaderProgram, 0, irradianceTexture);

	setUniformByName(*m_shaderProgram, "uWorldMatrix", worldMatrix);
	setUniformByName(*m_shaderProgram, "uViewProjMatrix", shaderUniforms.viewProjMatrix);
	setUniformByName(*m_shaderProgram, "uExposure", shaderUniforms.exposure);

	setVertexBuffer(*m_shaderProgram, m_vertexBuffer, sizeof(Vertex));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);
	glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);
}
