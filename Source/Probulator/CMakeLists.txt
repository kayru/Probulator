add_library(Probulator
	Common.h
	Compute.cpp
	Compute.h
	DiscreteDistribution.h
	ExperimentAmbientCube.cpp
	ExperimentAmbientCube.h
	ExperimentHBasis.h
	ExperimentMonteCarlo.h
	ExperimentRNM.cpp
	ExperimentRNM.h
	Experiments.cpp
	Experiments.h
	ExperimentSG.h
	ExperimentSH.h
	HBasis.h
	Image.cpp
	Image.h
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
	target_link_libraries(Probulator ${OpenCL_LIBRARIES})
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR
   ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	find_library(tbb tbb)
	target_link_libraries(Probulator ${tbb})
endif()

target_link_libraries(Probulator stb)
