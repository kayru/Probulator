#include "SphericalGaussian.h"

namespace Probulator
{
	float sgIntegral(float lambda)
	{
		return fourPi * (0.5f - 0.5f*exp(-2.0f*lambda)) / lambda;
	}

	vec3 sgDot(const SphericalGaussian& a, const SphericalGaussian& b)
	{
		float dM = length(a.lambda*a.p + b.lambda*b.p);
		vec3 num = sinh(dM) * fourPi * a.mu * b.mu;
		float den = exp(a.lambda + b.lambda) * dM;
		return num / den;
	}

	float sgEvaluate(const vec3& p, float lambda, const vec3& v)
	{
		float dp = dot(v, p);
		return exp(lambda * (dp - 1.0f));
	}

	vec3 sgEvaluate(const SphericalGaussian& sg, const vec3& v)
	{
		return sg.mu * sgEvaluate(sg.p, sg.lambda, v);
	}

	SphericalGaussian sgCross(const SphericalGaussian& a, const SphericalGaussian& b)
	{
		vec3 pM = (a.lambda*a.p + b.lambda*b.p) / (a.lambda + b.lambda);
		float pMLength = length(pM);
		float lambdaM = a.lambda + b.lambda;

		SphericalGaussian r;
		r.p = pM / pMLength;
		r.lambda = lambdaM * pMLength;
		r.mu = a.mu * b.mu * exp(lambdaM * (pMLength - 1.0f));

		return r;
	}

	float sgFindMu(float targetLambda, float targetIntegral)
	{
		return targetIntegral / sgIntegral(targetLambda);
	}

	float sgFindMu(float targetLambda, float lambda, float mu)
	{
		float targetIntegral = sgIntegral(lambda)*mu;
		return sgFindMu(targetLambda, targetIntegral);
	}

}