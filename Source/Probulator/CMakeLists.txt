if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-std=c++11)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	add_definitions(-std=c++11 -stdlib=libc++)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	find_path(tbbinclude tbb)
	include_directories(${tbbinclude})
endif()

find_package(OpenCL REQUIRED)

include_directories(
	${OpenCL_INCLUDE_DIRS}
)

add_executable(Probulator
	Common.h
	Compute.h
	Compute.cpp
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

target_link_libraries(Probulator
	${OpenCL_LIBRARIES}
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	find_library(tbb tbb)
	target_link_libraries(Probulator ${tbb})
endif()