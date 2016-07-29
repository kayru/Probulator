#include "Model.h"
#include "Shaders.h"

#include <tiny_obj_loader.h>

Model::Model(const char* objFilename)
{
    const bool forceGenerateNormals = false;
    readObj(objFilename, forceGenerateNormals);
}

Model::Model(const ProceduralSphere& sphere)
{
	generateSphere(sphere);
}

Model::Model(const ProceduralPlane& plane)
{
	generatePlane(plane);
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
	}

	m_dimensions = m_boundsMax - m_boundsMin;
	m_center = (m_boundsMax + m_boundsMin) / 2.0f;

	createBuffers(vertices, indices);

	return m_valid;
}

void Model::createBuffers(const Vertex* vertices, u32 vertexCount, const u32* indices, u32 indexCount)
{
	m_indexCount = indexCount;

	glGenBuffers(1, &m_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(Vertex), vertices, GL_STATIC_DRAW);

	glGenBuffers(1, &m_indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(u32), indices, GL_STATIC_DRAW);
}

void Model::createBuffers(const std::vector<Vertex>& vertices, const std::vector<u32>& indices)
{
	createBuffers(vertices.data(), (u32)vertices.size(), indices.data(), (u32)indices.size());
}

void Model::generatePlane(const ProceduralPlane& plane)
{
	vec2 halfSize = plane.dimensions * 0.5f;

    m_boundsMin = vec3(-halfSize, 0.0f);
    m_boundsMax = vec3(halfSize, 0.0f);
    m_dimensions = m_boundsMax - m_boundsMin;
    m_center = vec3(0.0f, 0.0f, 0.0f);
    m_valid = true;

	auto makeVert = [](vec3 p, vec3 n, vec2 t)
	{
		Vertex v;
		v.position = p;
		v.normal = n;
		v.texCoord = t;
		return v;
	};

	Vertex vertices[4] =
	{
		makeVert(vec3(-halfSize.x, -halfSize.y, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec2(0.0f, 1.0f)),
		makeVert(vec3(+halfSize.x, -halfSize.y, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec2(1.0f, 1.0f)),
		makeVert(vec3(-halfSize.x, +halfSize.y, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec2(0.0f, 0.0f)),
		makeVert(vec3(+halfSize.x, +halfSize.y, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec2(1.0f, 0.0f)),
	};

	u32 indices[6] =
	{
		0, 1, 2,
		2, 1, 3
	};

	createBuffers(vertices, 4, indices, 6);
}

void Model::generateSphere(const ProceduralSphere& sphere)
{
	const u64 NumUSlices = sphere.numUSlices;
	const u64 NumVSlices = sphere.numVSlices;
	
    m_boundsMin = vec3(-1.0f, -1.0f, -1.0f);
    m_boundsMax = vec3(+1.0f, +1.0f, +1.0f);
    m_dimensions = m_boundsMax - m_boundsMin;
    m_center = vec3(0.0f, 0.0f, 0.0f);
    m_valid = true;

    std::vector<Vertex> sphereVerts;

    // Add the vert at the top
    Vertex vert;
    vert.position = vec3(0.0f, 0.0f, 1.0f);
    vert.normal = vec3(0.0f, 0.0f, 1.0f);

    sphereVerts.push_back(vert);

    // Add the rings
    for(u64 v = 0; v < NumVSlices - 1; ++v)
    {
        for(u64 u = 0; u < NumUSlices; ++u)
        {
            const float theta = ((v + 1.0f) / NumVSlices) * pi;
            const float phi = (float(u) / NumUSlices) * twoPi;

            vec3 pos;
            pos.x = std::sin(theta) * std::cos(phi);
            pos.y = std::sin(theta) * std::sin(phi);
            pos.z = std::cos(theta);
            vert.position = pos;
            vert.normal = pos;
            sphereVerts.push_back(vert);
        }
    }

    // Add the vert at the bottom
    const u32 lastVertIdx = u32(sphereVerts.size());
    vert.position = vec3(0.0f, 0.0f, -1.0f);
    vert.normal = vec3(0.0f, 0.0f, -1.0f);
    sphereVerts.push_back(vert);

    // Add the top ring of triangles
    std::vector<u32> sphereIndices;
    for(u32 u = 0; u < NumUSlices; ++u)
    {
        sphereIndices.push_back(0);
        sphereIndices.push_back(u + 1);

        if(u < NumUSlices - 1)
            sphereIndices.push_back(u + 2);
        else
            sphereIndices.push_back(1);
    }

    // Add the rest of the rings
    u32 prevRowStart = 1;
    u32 currRowStart = u32(NumUSlices + 1);
    for(u32 v = 1; v < NumVSlices - 1; ++v)
    {
        for(u32 u = 0; u < NumUSlices; ++u)
        {
            u32 nextBottom = currRowStart + u + 1;
            u32 nextTop = prevRowStart + u + 1;
            if(u == NumUSlices - 1)
            {
                nextBottom = currRowStart;
                nextTop = prevRowStart;
            }

            sphereIndices.push_back(prevRowStart + u);
            sphereIndices.push_back(currRowStart + u);
            sphereIndices.push_back(nextBottom);
            sphereIndices.push_back(nextBottom);
            sphereIndices.push_back(nextTop);
            sphereIndices.push_back(prevRowStart + u);
        }

        prevRowStart = currRowStart;
        currRowStart += u32(NumUSlices);
    }

    // Add the last ring at the bottom
    const u32 lastRingStart = u32(lastVertIdx - NumUSlices);
    for(u32 u = 0; u < NumUSlices; ++u)
    {
        sphereIndices.push_back(lastVertIdx);
        sphereIndices.push_back(lastRingStart + u);

        if(u < NumUSlices - 1)
            sphereIndices.push_back(lastRingStart + u + 1);
        else
            sphereIndices.push_back(lastRingStart);
    }

	createBuffers(sphereVerts, sphereIndices);
}

void Model::draw(
	const ShaderProgram& shaderProgram,
	const CommonShaderUniforms& shaderUniforms,
	const Texture& irradianceTexture,
	const mat4& worldMatrix)
{
	if (!m_valid)
		return;

	glUseProgram(shaderProgram.m_native);

	setTexture(shaderProgram, 0, irradianceTexture);

	setCommonUniforms(shaderProgram, shaderUniforms);
	setUniformByName(shaderProgram, "uWorldMatrix", worldMatrix);

	setVertexBuffer(shaderProgram, m_vertexBuffer, sizeof(Vertex));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);
	glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);
}
