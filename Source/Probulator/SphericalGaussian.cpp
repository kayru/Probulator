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

    // Stephen Hill [2016], https://mynameismjp.wordpress.com/2016/10/09/sg-series-part-3-diffuse-lighting-from-an-sg-light-source/
    vec3 sgIrradianceFitted(const SphericalGaussian& lightingLobe, const vec3& normal)
    {
        if(lightingLobe.lambda == 0.f)
            return lightingLobe.mu;
        
        const float muDotN = dot(lightingLobe.p, normal);
        const float lambda = lightingLobe.lambda;
        
        const float c0 = 0.36f;
        const float c1 = 1.0f / (4.0f * c0);
        
        float eml  = exp(-lambda);
        float em2l = eml * eml;
        float rl   = 1.f / lambda;
        
        float scale = 1.0f + 2.0f * em2l - rl;
        float bias  = (eml - em2l) * rl - em2l;
        
        float x  = sqrt(1.0f - scale);
        float x0 = c0 * muDotN;
        float x1 = c1 * x;
        
        float n = x0 + x1;
        
        float y = saturate(muDotN);
        if(abs(x0) <= x1)
            y = n * n / x;
        
        float result = scale * y + bias;
        
        return result * lightingLobe.mu * sgIntegral(lightingLobe.lambda);
    }
}
