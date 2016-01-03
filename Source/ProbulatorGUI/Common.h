#pragma once

#include <Probulator/Common.h>
using namespace Probulator;

class NonCopyable
{
public:
	NonCopyable(){}
	NonCopyable(const NonCopyable&) = delete;
	NonCopyable(NonCopyable&&) = delete;
	NonCopyable& operator=(const NonCopyable&) = delete;
	NonCopyable& operator=(NonCopyable&&) = delete;
};
