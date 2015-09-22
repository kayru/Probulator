#pragma once

#include "Math.h"

namespace Probulator
{
	// On-line variance calculation algorithm (Knuth / Welford)
	// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
	template <typename T>
	struct OnlineVariance
	{
		int n = 0;
		T mean = T(0);
		T M2 = T(0);

		void addSample(T x)
		{
			n++;
			T delta = x - mean;
			mean = mean + delta / (float)n;
			M2 = M2 +  delta * (x - mean);
		}

		T getVariance()
		{
			if (n < 2)
			{
				return T(0);
			}
			T variance = M2 / (float)(n - 1);
			return variance;
		}
	};
}
