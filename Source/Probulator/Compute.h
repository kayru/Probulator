#pragma once

#ifdef USE_OPENCL
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#include "CL/cl_ext.h"
#endif

#include <string>
#include <vector>

namespace Probulator
{
	struct ComputeDevice
	{
		cl_device_id id;

		std::string name;
		cl_bool available;
		cl_device_type type;
		cl_ulong localMemSize;
	};

	class ComputeKernel
	{
	public:

		ComputeKernel(const char* code, const char* name);
		~ComputeKernel();

		cl_kernel getKernel() const { return m_kernel; }
		size_t getWorkGroupSize() const { return m_workGroupSize; }

		
		template <typename... Args>
		void setKernelArgs(const Args&... args)
		{
			setKernelArgsInternal(0, args...);
		}

	private:

		template <typename T>
		void setKernelArgsInternal(int index, const T& x)
		{
			clSetKernelArg(m_kernel, index, sizeof(T), &x);
		}

		template <typename T, typename... Args>
		void setKernelArgsInternal(int index, const T& x, const Args&... args)
		{
			clSetKernelArg(m_kernel, index, sizeof(T), &x);
			setKernelArgsInternal(index + 1, args...);
		}

		cl_program m_program = nullptr;
		cl_kernel m_kernel = nullptr;
		size_t m_workGroupSize = 0;
	};

	extern cl_device_id g_computeDevice;
	extern cl_context g_computeContext;
	extern cl_command_queue g_computeQueue;
}
#endif

