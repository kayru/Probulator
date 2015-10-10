#include "Compute.h"

#include <stdio.h>
#include <assert.h>

namespace
{
	template <typename T>
	void getInfo(cl_device_id id, cl_device_info info, T& outVal);

	template <typename T>
	void getInfo(cl_device_id id, cl_device_info info, T& outVal)
	{
		clGetDeviceInfo(id, info, sizeof(T), &outVal, nullptr);
	}

	template <>
	void getInfo(cl_device_id id, cl_device_info info, std::string& outVal)
	{
		size_t valSize;
		clGetDeviceInfo(id, info, 0, nullptr, &valSize);
		outVal.resize(valSize);
		clGetDeviceInfo(id, info, valSize, &outVal[0], nullptr);
	}

	template <typename T>
	void getInfo(cl_device_id id, cl_device_info info, std::vector<T>& outVal)
	{
		size_t valSize;
		clGetDeviceInfo(id, info, 0, nullptr, &valSize);
		outVal.resize(valSize / sizeof(T));
		clGetDeviceInfo(id, info, valSize, &outVal[0], nullptr);
	}
}

namespace Probulator
{
	cl_device_id g_computeDevice;
	cl_context g_computeContext;
	cl_command_queue g_computeQueue;

	static std::vector<ComputeDevice> g_devices;

	struct ComputeDeviceInitializer
	{
		ComputeDeviceInitializer()
		{
			cl_uint platformIdCount = 0;
			clGetPlatformIDs(0, nullptr, &platformIdCount);

			std::vector<cl_platform_id> platformIds(platformIdCount);
			clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

			cl_uint deviceIdCount = 0;
			clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceIdCount);

			std::vector<cl_device_id> deviceIds(deviceIdCount);
			clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), nullptr);

			g_devices.resize(deviceIdCount);
			for (cl_uint i = 0; i < deviceIdCount; ++i)
			{
				cl_device_id id = deviceIds[i];
				ComputeDevice& device = g_devices[i];

				device.id = id;
				getInfo(id, CL_DEVICE_NAME, device.name);
				getInfo(id, CL_DEVICE_AVAILABLE, device.available);
				getInfo(id, CL_DEVICE_TYPE, device.type);
			}

			const cl_context_properties contextProperties[] =
			{
				CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platformIds[0]),
				0, 0
			};

			cl_int error = CL_SUCCESS;

			g_computeContext = clCreateContext(contextProperties, deviceIdCount, deviceIds.data(), nullptr, nullptr, &error);
			assert(error == CL_SUCCESS);

			size_t bestDeviceIndex = 0;
			for (size_t deviceIt = 0; deviceIt < g_devices.size(); ++deviceIt)
			{
				if (g_devices[deviceIt].type == CL_DEVICE_TYPE_GPU)
				{
					bestDeviceIndex = deviceIt;
					break;
				}
			}

			g_computeQueue = clCreateCommandQueue(g_computeContext, g_devices[bestDeviceIndex].id, 0, &error);
			g_computeDevice = g_devices[bestDeviceIndex].id;
			assert(error == CL_SUCCESS && "Failed to create command queue");

		}

		~ComputeDeviceInitializer()
		{
			clReleaseCommandQueue(g_computeQueue);
			clReleaseContext(g_computeContext);
			g_devices.clear();
		}
	};
	static ComputeDeviceInitializer g_computeInitializer;

	ComputeKernel::ComputeKernel(const char* code, const char* name)
	{
		cl_int error = CL_SUCCESS;

		m_program = clCreateProgramWithSource(g_computeContext, 1, (const char**)&code, nullptr, &error);
		assert(error == CL_SUCCESS && "clCreateProgramWithSource failed");

		error = clBuildProgram(m_program, 0, nullptr, nullptr, nullptr, nullptr);
		if (error != CL_SUCCESS)
		{
			size_t len = 0;
			char errorBuffer[2048] = {};
			clGetProgramBuildInfo(m_program, g_computeDevice, CL_PROGRAM_BUILD_LOG, sizeof(errorBuffer), errorBuffer, &len);
			printf("Failed to build compute kernel '%s': %s\n", name, errorBuffer);
		}

		m_kernel = clCreateKernel(m_program, name, &error);
		assert(error == CL_SUCCESS && "clCreateKernel failed");

		error = clGetKernelWorkGroupInfo(m_kernel, g_computeDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(m_workGroupSize), &m_workGroupSize, nullptr);
		assert(error == CL_SUCCESS && "clGetKernelWorkGroupInfo failed");
	}

	ComputeKernel::~ComputeKernel()
	{
		clReleaseKernel(m_kernel);
		clReleaseProgram(m_program);
	}
}
