#include "Common.glsl"

in vec2 Position;
in vec2 TexCoord0;

out vec2 vTexCoord0;
out vec4 vPosition;

void main()
{
	vPosition = vec4(Position.x, Position.y, 1.0, 1.0);
	vTexCoord0 = Position * 0.5 + 0.5;
	vTexCoord0.y = 1.0 - vTexCoord0.y;
	gl_Position = vPosition;
}
