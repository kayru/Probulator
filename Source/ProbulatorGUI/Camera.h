#pragma once

#include "Common.h"
#include <Probulator/Math.h>

struct Camera
{
	mat4 getProjectionMatrix() const;
	mat4 getViewMatrix() const;

	void rotate(float deltaAroundUp, float deltaAroundRight);
	void move(const vec3& delta);

	float m_aspect = 1.0f;
	float m_fov = 1.0f;
	float m_near = 1.0f;
	float m_far = 1000.0f;
	mat3 m_orientation = mat3(
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f);
	vec3 m_position = vec3(0.0f);
};

enum CameraMode
{
	CameraMode_FirstPerson,
	CameraMode_Orbit,

	CameraModeCount
};

const char* toString(CameraMode mode);

struct CameraController
{
	struct InputState
	{
		float moveForward = 0.0f;
		float moveUp = 0.0f;
		float moveRight = 0.0f;
		float rotateAroundUp = 0.0f;
		float rotateAroundRight = 0.0f;
		float moveSpeedMultiplier = 1.0f;
		float rotateSpeedMultiplier = 1.0f;
	};

	float m_rotateSpeed = 0.01f;
	float m_moveSpeed = 0.05f;

	vec3 m_orbitCenter = vec3(0.0f, 0.0f, 0.0f);
	float m_orbitRadius = 0.0f;

	CameraMode m_mode = CameraMode_FirstPerson;

	void update(const InputState& input, Camera& camera);

	Camera interpolate(const Camera& x, const Camera& y, 
		float positionAlpha, float orientationAlpha, float attributeAlpha = 1.0f);
};

