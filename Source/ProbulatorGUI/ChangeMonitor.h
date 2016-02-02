#pragma once

#include "Common.h"

class ChangeMonitor
{
public:
	virtual ~ChangeMonitor() {}
	virtual bool update() = 0;
};

ChangeMonitor* createChangeMonitor(const char* path);

