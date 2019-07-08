#include "Thread.h"

namespace Probulator
{
	enki::TaskScheduler g_TS;

	struct TaskSchedulerInitializer
	{
		TaskSchedulerInitializer()
		{
			g_TS.Initialize();
		}
	};
	
	static TaskSchedulerInitializer taskSchedulerInitializer;
}