#include <Probulator/Experiments.h>

namespace Probulator
{

template <typename T>
static T& addExperiment(ExperimentList& list, const char* name, const char* suffix)
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

    addExperiment<ExperimentMCIS>(experiments, "Monte Carlo [Importance Sampling]", "MCIS")
        .setSampleCount(5000)
        .setScramblingEnabled(false) // prefer errors due to correlation instead of noise due to scrambling
        .setUseAsReference(true); // other experiments will be compared against this

    addExperiment<ExperimentMCIS>(experiments, "Monte Carlo [Importance Sampling, Scrambled]", "MCISS")
        .setSampleCount(5000)
        .setScramblingEnabled(true)
        .setEnabled(false); // disabled by default, since MCIS mode is superior

    addExperiment<ExperimentMC>(experiments, "Monte Carlo", "MC")
        .setHemisphereSampleCount(5000)
        .setEnabled(false); // disabled by default, since MCIS mode is superior

    addExperiment<ExperimentSHL1Geomerics>(experiments, "Spherical Harmonics L1 [Geomerics]", "SHL1G");

    addExperiment<ExperimentSH<1>>(experiments, "Spherical Harmonics L1", "SHL1");
    addExperiment<ExperimentSH<2>>(experiments, "Spherical Harmonics L2", "SHL2");
    addExperiment<ExperimentSH<3>>(experiments, "Spherical Harmonics L3", "SHL3");
    addExperiment<ExperimentSH<4>>(experiments, "Spherical Harmonics L4", "SHL4");

    addExperiment<ExperimentHBasis<4>>(experiments, "HBasis-4", "H4");
    addExperiment<ExperimentHBasis<6>>(experiments, "HBasis-6", "H6");

    addExperiment<ExperimentSGNaive>(experiments, "Spherical Gaussians [Naive]", "SG")
        .setBrdfLambda(8.5f) // Chosen arbitrarily through experimentation
        .setLobeCountAndLambda(lobeCount, lambda);

    addExperiment<ExperimentSGLS>(experiments, "Spherical Gaussians [Least Squares]", "SGLS")
        .setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
        .setLobeCountAndLambda(lobeCount, lambda);

    addExperiment<ExperimentSGLS>(experiments, "Spherical Gaussians [Least Squares + Ambient]", "SGLSA")
        .setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
        .setAmbientLobeEnabled(true)
        .setLobeCountAndLambda(lobeCount, lambda);

    addExperiment<ExperimentSGNNLS>(experiments, "Spherical Gaussians [Non-Negative Least Squares]", "SGNNLS")
        .setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
        .setLobeCountAndLambda(lobeCount, lambda);

    addExperiment<ExperimentSGGA>(experiments, "Spherical Gaussians [Genetic Algorithm]", "SGGA")
        .setPopulationAndGenerationCount(50, 2000)
        .setBrdfLambda(3.0f) // Chosen arbitrarily through experimentation
        .setLobeCountAndLambda(lobeCount, lambda)
        .setEnabled(false); // disabled by default, as it requires *very* long time to converge
}

} // namespace Probulator