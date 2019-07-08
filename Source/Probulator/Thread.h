#pragma once

#include <TaskScheduler.h>
#include <thread>

namespace Probulator
{
	extern enki::TaskScheduler g_TS;

	template <typename I, typename F>
	inline void parallelFor(I begin, I end, F fun)
	{
#if 1
		enki::TaskSet taskSet(end - begin,
			[&](enki::TaskSetPartition range, I threadnum)
		{
			for(I i=range.start; i!=range.end; ++i)
			{
				fun(i);
			}
		});
		g_TS.AddTaskSetToPipe(&taskSet);
		g_TS.WaitforTask(&taskSet);
#else
		for(I i=begin; i!=end; ++i)
		{
			fun(i);
		}
#endif
	}
}
