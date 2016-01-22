add_library(Probulator
	Compute.cpp
	Experiments.cpp
	Image.cpp
	SGBasis.cpp
	SGFitGeneticAlgorithm.cpp
	SGFitLeastSquares.cpp
	SphericalGaussian.cpp
	Common.h
	Compute.h
	DiscreteDistribution.h
	ExperimentHBasis.h
	ExperimentMonteCarlo.h
	Experiments.h
	ExperimentSG.h
	ExperimentSH.h
	HBasis.h
	Image.h
	Math.h
	RadianceSample.h
	SGBasis.h
	SGFitGeneticAlgorithm.h
	SGFitLeastSquares.h
	SphericalGaussian.h
	SphericalHarmonics.h
	Thread.h
	Variance.h
)

if(OpenCL_FOUND)
	target_link_libraries(Probulator ${OpenCL_LIBRARIES})
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR
   ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	find_library(tbb tbb)
	target_link_libraries(Probulator ${tbb})
endif()

target_link_libraries(Probulator stb)
