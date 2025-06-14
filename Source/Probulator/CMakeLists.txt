add_library(Probulator
	Experiments.cpp
	Image.cpp
	SGBasis.cpp
	SGFitGeneticAlgorithm.cpp
	SGFitLeastSquares.cpp
	SphericalGaussian.cpp
	Common.h
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
	ExperimentZH3.h
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

target_link_libraries(Probulator stb enkiTS glm eigen lbfgs zh3solver)
target_compile_features(Probulator PUBLIC cxx_std_11)
target_include_directories(Probulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
