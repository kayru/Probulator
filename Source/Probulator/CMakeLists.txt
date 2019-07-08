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
	ExperimentAmbientCube.h
	ExperimentAmbientCube.cpp
	ExperimentAmbientDice.h
	ExperimentAmbientDice.cpp
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
	Thread.cpp
	Variance.h
)

if(OpenCL_FOUND)
	target_link_libraries(Probulator ${OpenCL_LIBRARIES})
endif()

target_link_libraries(Probulator stb enkiTS)
