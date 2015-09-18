#pragma once

#ifdef _MSC_VER
#include <ppl.h>
#include <ppltasks.h>
#else
#include <tbb/compat/ppl.h>
#endif

#include <thread>

namespace Probulator
{
	template <typename I, typename F>
	inline void parallelFor(I begin, I end, F fun)
	{
		Concurrency::parallel_for<I>(begin, end, fun);
	}
}
