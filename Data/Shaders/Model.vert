#include "Common.glsl"

uniform mat4 uWorldMatrix;
uniform mat4 uViewProjMatrix;

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord0;

out vec3 vWorldNormal;
out vec2 vTexCoord0;
out vec3 vWorldPosition;

void main()
{
	vec3 worldPosition = vec3(uWorldMatrix * vec4(Position, 1));
	vec3 worldNormal = normalize(vec3(mat3(uWorldMatrix) * Normal));
	gl_Position = uViewProjMatrix * vec4(worldPosition, 1);

	vWorldPosition = worldPosition;
	vWorldNormal = worldNormal;
	vTexCoord0 = TexCoord0;
}
