#include "Common.glsl"

uniform mat4 uWorldMatrix;
uniform mat4 uViewProjMatrix;

uniform sampler2D Texture0;

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

    vec2 texCoord = cartesianToLatLongTexcoord(worldNormal);
    vec3 irradiance = texture(Texture0, texCoord).xyz;
    float lumScale = 0.299 * irradiance.r + 0.587 * irradiance.g + 0.114 * irradiance.b;

	gl_Position = uViewProjMatrix * vec4(worldPosition * lumScale, 1);

	vWorldPosition = worldPosition;
	vWorldNormal = worldNormal;
	vTexCoord0 = TexCoord0;
}
