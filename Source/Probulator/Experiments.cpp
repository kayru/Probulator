#include <Probulator/Experiments.h>

#include <Probulator/ExperimentMonteCarlo.h>
#include <Probulator/ExperimentSH.h>
#include <Probulator/ExperimentSG.h>
#include <Probulator/ExperimentHBasis.h>
#include <Probulator/ExperimentAmbientCube.h>
#include <Probulator/ExperimentAmbientDice.h>
#include <Probulator/ExperimentZH3.h>

namespace Probulator
{

template <typename T> 
inline T& addExperiment(ExperimentList& list, const char* name, const char* suffix)
{
    T* e = new T;
    list.push_back(std::unique_ptr<Experiment>(e));
    e->m_name = name;
    e->m_suffix = suffix;
    return *e;
}

void addAllExperiments(ExperimentList& experiments)
{
    const u32 lobeCount = 12; // <-- tweak this
    const float lambda = 0.5f * lobeCount; // <-- tweak this; 

    Experiment* experimentMCIS = &addExperiment<ExperimentMCIS>(experiments, "Monte Carlo [Importance Sampling]", "MCIS")
        .setSampleCount(5000)
        .setJitterEnabled(false) // prefer errors due to correlation instead of noise due to jittering
        .setUseAsReference(true); // other experiments will be compared against this

    addExperiment<ExperimentMCIS>(experiments, "Monte Carlo [Importance Sampling, Jittered]", "MCISS")
        .setSampleCount(5000)
        .setJitterEnabled(true)
        .setEnabled(false); // disabled by default, since MCIS mode is superior

    addExperiment<ExperimentMC>(experiments, "Monte Carlo", "MC")
        .setHemisphereSampleCount(5000)
        .setEnabled(false); // disabled by default, since MCIS mode is superior

	addExperiment<ExperimentAmbientCube>(experiments, "Ambient Cube [Non-Negative Least Squares]", "AC")
		.setProjectionEnabled(false)
		.setInput(experimentMCIS);

	addExperiment<ExperimentAmbientCube>(experiments, "Ambient Cube [Projection]", "ACPROJ")
		.setProjectionEnabled(true)
		.setInput(experimentMCIS);

    addExperiment<ExperimentSHL1Geomerics>(experiments, "Spherical Harmonics L1 [Geomerics]", "SHL1G");

    addExperiment<ExperimentZH3>(experiments, "ZH3 [Solve]", "ZH3");
    addExperiment<ExperimentHallucinateZH3>(experiments, "ZH3 [Hallucinate from Spherical Harmonics L1]", "ZH3H");
    addExperiment<ExperimentSolveHallucinateZH3>(experiments, "ZH3 [Hallucinate from solved Spherical Harmonics L1]", "ZH3HS");

    addExperiment<ExperimentSH<1>>(experiments, "Spherical Harmonics L1", "SHL1");
    addExperiment<ExperimentSH<2>>(experiments, "Spherical Harmonics L2", "SHL2");
    addExperiment<ExperimentSH<3>>(experiments, "Spherical Harmonics L3", "SHL3");
    addExperiment<ExperimentSH<4>>(experiments, "Spherical Harmonics L4", "SHL4");

	addExperiment<ExperimentSH<2>>(experiments, "Spherical Harmonics L2 [Windowed]", "SHL2W")
		.setTargetLaplacian(10.0f); // Empirically chosen

    addExperiment<ExperimentHBasis<4>>(experiments, "HBasis-4", "H4")
        .setInput(experimentMCIS)
		.setEnabled(false);

    addExperiment<ExperimentHBasis<6>>(experiments, "HBasis-6", "H6")
        .setInput(experimentMCIS)
		.setEnabled(false);
    
    addExperiment<ExperimentAmbientDice>(experiments, "Ambient Dice [Bezier]", "AD")
    .setDiceType(AmbientDiceTypeBezier)
    .setInput(experimentMCIS);
    
    addExperiment<ExperimentAmbientDice>(experiments, "Ambient Dice [Bezier Y/Co/Cg]", "ADYCoCg")
    .setDiceType(AmbientDiceTypeBezierYCoCg)
    .setInput(experimentMCIS);
    
    addExperiment<ExperimentAmbientDice>(experiments, "Ambient Dice [Cosine SRBF]", "ADRBF")
    .setDiceType(AmbientDiceTypeSRBF)
    .setInput(experimentMCIS);

    addExperiment<ExperimentSGNaive>(experiments, "Spherical Gaussians [Naive]", "SG")
        .setBrdfLambda(8.5f) // Chosen arbitrarily through experimentation
        .setLobeCountAndLambda(lobeCount, lambda);

    addExperiment<ExperimentSGLS>(experiments, "Spherical Gaussians [Least Squares]", "SGLS")
        .setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
        .setLobeCountAndLambda(lobeCount, lambda);

    addExperiment<ExperimentSGLS>(experiments, "Spherical Gaussians [Least Squares + Ambient]", "SGLSA")
        .setAmbientLobeEnabled(true)
        .setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
        .setLobeCountAndLambda(lobeCount, lambda);

    addExperiment<ExperimentSGNNLS>(experiments, "Spherical Gaussians [Non-Negative Least Squares]", "SGNNLS")
        .setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
        .setLobeCountAndLambda(lobeCount, lambda);
    
    addExperiment<ExperimentSGRunningAverage>(experiments, "Spherical Gaussians [Running Average]", "SGRA")
        .setLobeCountAndLambda(lobeCount, lambda);
    
    addExperiment<ExperimentSGRunningAverage>(experiments, "Spherical Gaussians [Non-Negative Running Average]", "SGNNRA")
        .setLobeCountAndLambda(lobeCount, lambda)
        .setNonNegativeSolve(true);

    addExperiment<ExperimentSGGA>(experiments, "Spherical Gaussians [Genetic Algorithm]", "SGGA")
        .setPopulationAndGenerationCount(50, 2000)
        .setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
        .setLobeCountAndLambda(lobeCount, lambda)
        .setEnabled(false); // disabled by default, as it requires *very* long time to converge
}

void resetAllExperiments(ExperimentList& experiments)
{
	for (auto& e : experiments)
	{
		e->reset();
	}
}

} // namespace Probulator
