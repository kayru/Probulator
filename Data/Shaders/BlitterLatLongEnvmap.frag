#include "Common.glsl"

uniform mat4 uViewMatrix;
uniform mat4 uProjMatrix;
uniform sampler2D Texture0;

in vec4 vPosition;

out vec4 Target;

void main()
{
	vec3 view;
	view.x = vPosition.x / uProjMatrix[0][0];
	view.y = vPosition.y / uProjMatrix[1][1];
	view.z = -1.0;
	view = normalize(view * mat3(uViewMatrix));
	vec2 texCoord = cartesianToLatLongTexcoord(view);
	vec4 color = texture(Texture0, texCoord);
	color.rgb = tonemapLinear(color.rgb, uExposure);
	color.rgb = applyDithering(color.rgb, gl_FragCoord.xy / uResolution, uElapsedTime);
	Target = color;
}
