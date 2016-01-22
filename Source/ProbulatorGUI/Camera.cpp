#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>

Probulator::mat4 Camera::getProjectionMatrix() const
{
	return glm::perspective(m_fov, m_aspect, m_near, m_far);
}

mat4 Camera::getViewMatrix() const
{
	vec3 eyePosition = m_position + m_orbitRadius * m_orientation[2];

	vec3 p;
	p.x = -dot((vec3)m_orientation[0], eyePosition);
	p.y = -dot((vec3)m_orientation[1], eyePosition);
	p.z = -dot((vec3)m_orientation[2], eyePosition);

	mat3 r = transpose(m_orientation);

	return mat4(
		vec4(r[0], 0.0),
		vec4(r[1], 0.0),
		vec4(r[2], 0.0),
		vec4(p, 1.0f));
}

void Camera::rotate(float deltaAroundUp, float deltaAroundRight)
{
	mat3 rotUp = (mat3)glm::rotate(-deltaAroundUp, vec3(0.0f, 1.0f, 0.0f));
	mat3 rotRight = (mat3)glm::rotate(-deltaAroundRight, m_orientation[0]);
	m_orientation = rotUp * rotRight * m_orientation;
}

void Camera::moveViewSpace(const vec3& viewSpaceDelta)
{
	vec3 worldSpaceDelta = m_orientation * viewSpaceDelta;
	m_position += worldSpaceDelta;
}

void Camera::moveWorldSpace(const vec3& worldSpaceDelta)
{
	m_position += worldSpaceDelta;
}

void CameraController::update(const InputState& input, Camera& camera)
{
	float moveSpeed = m_moveSpeed * input.moveSpeedMultiplier;
	float rotateSpeed = m_rotateSpeed * input.rotateSpeedMultiplier;

	camera.rotate(
		input.rotateAroundUp * rotateSpeed,
		input.rotateAroundRight * rotateSpeed);

	vec3 viewSpaceDelta = vec3(
		input.moveRight,
		input.moveUp,
		-input.moveForward);

	if (m_mode == CameraMode_Orbit)
	{
		if (camera.m_orbitRadius == 0.0f)
		{
			camera.moveViewSpace(vec3(0.0f, 0.0f, -m_orbitRadius));
		}

		m_orbitRadius = max(0.0f, m_orbitRadius - input.scrollDelta.y * 0.25f);

		camera.m_orbitRadius = m_orbitRadius;
	}
	else
	{
		if (camera.m_orbitRadius != 0.0f)
		{
			camera.moveViewSpace(vec3(0.0f, 0.0f, m_orbitRadius));
		}

		camera.m_orbitRadius = 0.0f;
	}

	float viewSpaceDeltaLength = length(viewSpaceDelta);
	if (viewSpaceDeltaLength > 0.0f)
	{
		camera.moveViewSpace(moveSpeed * viewSpaceDelta / viewSpaceDeltaLength);
	}
}

const char* toString(CameraMode mode)
{
	switch (mode)
	{
	default:
		assert(false && "Unknown camera mode");
		return "Unknown";
	case CameraMode_FirstPerson:
		return "FirstPerson";
	case CameraMode_Orbit:
		return "Orbit";
	}
}

Camera CameraController::interpolate(const Camera& x, const Camera& y, 
	float positionAlpha, float orientationAlpha, float attributeAlpha)
{
	Camera result;

	result.m_aspect = mix(x.m_aspect, y.m_aspect, attributeAlpha);
	result.m_fov = mix(x.m_fov, y.m_fov, attributeAlpha);
	result.m_near = mix(x.m_near, y.m_near, attributeAlpha);
	result.m_far = mix(x.m_far, y.m_far, attributeAlpha);
	result.m_position = mix(x.m_position, y.m_position, positionAlpha);
	result.m_orbitRadius = mix(x.m_orbitRadius, y.m_orbitRadius, positionAlpha);

	glm::quat qx = (glm::quat)x.m_orientation;
	glm::quat qy = (glm::quat)y.m_orientation;
	glm::quat qr = normalize(glm::slerp(qx, qy, orientationAlpha));
	
	result.m_orientation = (mat3)qr;

	return result;
}
