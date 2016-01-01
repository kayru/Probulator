if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-std=c++11)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	add_definitions(-std=c++11 -stdlib=libc++)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR
   ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	find_path(tbbinclude tbb)
	include_directories(${tbbinclude})
endif()

find_package(OpenCL)

if(OpenCL_FOUND)
	add_definitions(-DUSE_OPENCL)
	include_directories(
		${OpenCL_INCLUDE_DIRS}
	)
endif()

add_executable(Probulator
	Common.h
	Compute.cpp
	Compute.h
	DiscreteDistribution.h
	Image.cpp
	Image.h
	Main.cpp
	Math.h
	RadianceSample.h
	SGBasis.cpp
	SGBasis.h
	SGFitGeneticAlgorithm.cpp
	SGFitGeneticAlgorithm.h
	SGFitLeastSquares.cpp
	SGFitLeastSquares.h
	SphericalGaussian.cpp
	SphericalGaussian.h
	SphericalHarmonics.h
	Thread.h
	Variance.h
)

if(OpenCL_FOUND)
	target_link_libraries(Probulator
		${OpenCL_LIBRARIES}
)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR
   ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	find_library(tbb tbb)
	target_link_libraries(Probulator ${tbb})
endif()

