add_subdirectory(External)

if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-std=c++11)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	add_definitions(-std=c++11 -stdlib=libc++)
endif()

include_directories(
	.
	External/glm
	External/stb
	External/eigen
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR
   ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	find_path(tbbinclude tbb)
	include_directories(${tbbinclude})
endif()

find_package(OpenCL)

if(OpenCL_FOUND)
	add_definitions(-DUSE_OPENCL)
	include_directories(${OpenCL_INCLUDE_DIRS})
endif()

add_subdirectory(Probulator)
add_subdirectory(ProbulatorCLI)

include_directories(
	External/gl3w/include
	External/glfw/include
	External/imgui
)
add_subdirectory(ProbulatorGUI)
