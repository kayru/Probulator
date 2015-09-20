#include "SphericalHarmonics.h"

namespace Probulator
{
	// https://graphics.stanford.edu/papers/envmap/envmap.pdf

	SphericalHarmonicsL2 shEvaluateL2(vec3 p)
	{
		float c1 = 0.282095f;
		float c2 = 0.488603f;
		float c3 = 1.092548f;
		float c4 = 0.315392f;
		float c5 = 0.546274f;

		SphericalHarmonicsL2 sh;

		sh[0] = c1; // Y00

		sh[1] = c2 * p.y; // Y1-1
		sh[2] = c2 * p.z; // Y10
		sh[3] = c2 * p.x; // Y1+1

		sh[4] = c3 * p.x * p.y; // Y2-2
		sh[5] = c3 * p.y * p.z; // Y2-1
		sh[6] = c4 * (3.0f * p.z * p.z - 1.0f); // Y20
		sh[7] = c3 * p.x * p.z; // Y2+1
		sh[8] = c5 * (p.x * p.x - p.y * p.y); // Y2+2

		return sh;
	}

	float shEvaluateDiffuseL2(const SphericalHarmonicsL2& sh, vec3 n)
	{
		float c1 = 0.429043f;
		float c2 = 0.511664f;
		float c3 = 0.743125f;
		float c4 = 0.886227f;
		float c5 = 0.247708f;
		
		// l = 0, m = 0
		float L00 = sh[0];

		// l = 1, m= -1 0 +1 
		float L1_1 = sh[1];
		float L10 = sh[2];
		float L11 = sh[3];

		// l = 2, m= -2 -1 0 +1 +2
		float L2_2 = sh[4];
		float L2_1 = sh[5];
		float L20 = sh[6];
		float L21 = sh[7];
		float L22 = sh[8];

		float x = n.x;
		float y = n.y;
		float z = n.z;

		float x2 = n.x * n.x;
		float y2 = n.y * n.y;
		float z2 = n.z * n.z;

		return c1*L22*(x2 - y2) 
			+ c3*L20*z2 
			+ c4*L00 
			- c5*L20
			+ 2.0f * c1 * (L2_2*x*y + L21*x*z + L2_1*y*z)
			+ 2.0f * c2 * (L11*x + L1_1*y + L10*z);
	}

	void shReduceRingingL2(SphericalHarmonicsL2& sh, float lambda)
	{
		// From Peter-Pike Sloan's Stupid SH Tricks
		// http://www.ppsloan.org/publications/StupidSH36.pdf

		auto scale = [lambda](u32 L) { return 1.0f / (1.0f + lambda * L * L * (L + 1.0f) * (L + 1.0f)); };
		
		sh[0] *= scale(0);

		sh[1] *= scale(1);
		sh[2] *= scale(1);
		sh[3] *= scale(1);

		sh[4] *= scale(2);
		sh[5] *= scale(2);
		sh[6] *= scale(2);
		sh[7] *= scale(2);
		sh[8] *= scale(2);
	}

}
