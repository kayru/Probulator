Probulator
----------

Experimentation framework for probe-based lighting.

![ProbulatorGUI_Armadillo](https://github.com/kayru/Probulator/raw/master/Screenshots/ProbulatorGUI_Armadillo.jpg)

Example report
--------------

| Radiance | Irradiance  | Irradiance Error (sMAPE) | Mode
| --- | ---  | --- | ---
| ![Radiance] [Ennis-radianceMCIS.png]| ![Irradiance] [Ennis-irradianceMCIS.png] | **N/A** | Monte Carlo <br>[Importance Sampling]<br>**Reference**
| ![Radiance] [Ennis-radianceAC.png]| ![Irradiance] [Ennis-irradianceAC.png]| ![IrradianceError] [Ennis-irradianceErrorAC.png] | Ambient Cube
| ![Radiance] [Ennis-radianceSHL1G.png]| ![Irradiance] [Ennis-irradianceSHL1G.png]| ![IrradianceError] [Ennis-irradianceErrorSHL1G.png] | Spherical Harmonics L1 <br>[Geomerics]
| ![Radiance] [Ennis-radianceSHL1.png]| ![Irradiance] [Ennis-irradianceSHL1.png]| ![IrradianceError] [Ennis-irradianceErrorSHL1.png] | Spherical Harmonics L1
| ![Radiance] [Ennis-radianceSHL2.png]| ![Irradiance] [Ennis-irradianceSHL2.png]| ![IrradianceError] [Ennis-irradianceErrorSHL2.png] | Spherical Harmonics L2
| ![Radiance] [Ennis-radianceSHL3.png]| ![Irradiance] [Ennis-irradianceSHL3.png]| ![IrradianceError] [Ennis-irradianceErrorSHL3.png] | Spherical Harmonics L3
| ![Radiance] [Ennis-radianceSHL4.png]| ![Irradiance] [Ennis-irradianceSHL4.png]| ![IrradianceError] [Ennis-irradianceErrorSHL4.png] | Spherical Harmonics L4
| ![Radiance] [Ennis-radianceSHL2W.png]| ![Irradiance] [Ennis-irradianceSHL2W.png]| ![IrradianceError] [Ennis-irradianceErrorSHL2W.png] | Spherical Harmonics L2 <br>[Windowed]
| ![Radiance] [Ennis-radianceSG.png]| ![Irradiance] [Ennis-irradianceSG.png]| ![IrradianceError] [Ennis-irradianceErrorSG.png] | Spherical Gaussians <br>[Naive]
| ![Radiance] [Ennis-radianceSGLS.png]| ![Irradiance] [Ennis-irradianceSGLS.png]| ![IrradianceError] [Ennis-irradianceErrorSGLS.png] | Spherical Gaussians <br>[Least Squares]
| ![Radiance] [Ennis-radianceSGLSA.png]| ![Irradiance] [Ennis-irradianceSGLSA.png]| ![IrradianceError] [Ennis-irradianceErrorSGLSA.png] | Spherical Gaussians <br>[Least Squares + Ambient]
| ![Radiance] [Ennis-radianceSGNNLS.png]| ![Irradiance] [Ennis-irradianceSGNNLS.png]| ![IrradianceError] [Ennis-irradianceErrorSGNNLS.png] | Spherical Gaussians <br>[Non-Negative Least Squares]

Other pre-generated reports in HTML and Markdown formats are [included in the repository](https://github.com/kayru/Probulator/tree/master/Reports).

* [Ennis](https://github.com/kayru/Probulator/blob/master/Reports/Ennis/report.md)
* [Grace](https://github.com/kayru/Probulator/blob/master/Reports/Grace/report.md)
* [Pisa](https://github.com/kayru/Probulator/blob/master/Reports/Pisa/report.md)
* [Uffizi](https://github.com/kayru/Probulator/blob/master/Reports/Uffizi/report.md)
* [Wells](https://github.com/kayru/Probulator/blob/master/Reports/Wells/report.md)

How to build
------------

CMake is used to generate native build projects for target platform. 

#### Windows, Visual Studio 2015 ####

	mkdir Build
	cd Build
	cmake -G "Visual Studio 14 2015 Win64" ..
	cmake --build . --config Release

Visual Studio 2013 may also work.

#### MacOS, Xcode ####

	mkdir Build
	cd Build
	cmake -G Xcode ..
	cmake --build . --config Release

#### Linux ####

Linux support is not fully implemented.

How to run
----------

Run **ProbulatorGUI** from the build output directory (`Build/Source/ProbulatorGUI/Release` by default).

HDR Probe Credits
-----------------

[Bernhard Vogl](http://dativ.at/lightprobes)

 - wells.hdr

[USC Institute for Creative Technologies](http://gl.ict.usc.edu/Data/HighResProbes)

 - ennis.hdr
 - grace.hdr
 - pisa.hdr
 - uffizi.hdr
 
 Authors
 -------

 * Yuriy O'Donnell
 * David Neubelt

[Ennis-radianceMCIS.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceMCIS.png
[Ennis-irradianceMCIS.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceMCIS.png
[Ennis-radianceAC.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceAC.png
[Ennis-irradianceAC.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceAC.png
[Ennis-irradianceErrorAC.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorAC.png
[Ennis-radianceSHL1G.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSHL1G.png
[Ennis-irradianceSHL1G.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSHL1G.png
[Ennis-irradianceErrorSHL1G.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSHL1G.png
[Ennis-radianceSHL1.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSHL1.png
[Ennis-irradianceSHL1.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSHL1.png
[Ennis-irradianceErrorSHL1.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSHL1.png
[Ennis-radianceSHL2.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSHL2.png
[Ennis-irradianceSHL2.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSHL2.png
[Ennis-irradianceErrorSHL2.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSHL2.png
[Ennis-radianceSHL3.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSHL3.png
[Ennis-irradianceSHL3.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSHL3.png
[Ennis-irradianceErrorSHL3.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSHL3.png
[Ennis-radianceSHL4.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSHL4.png
[Ennis-irradianceSHL4.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSHL4.png
[Ennis-irradianceErrorSHL4.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSHL4.png
[Ennis-radianceSHL2W.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSHL2W.png
[Ennis-irradianceSHL2W.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSHL2W.png
[Ennis-irradianceErrorSHL2W.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSHL2W.png
[Ennis-radianceSG.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSG.png
[Ennis-irradianceSG.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSG.png
[Ennis-irradianceErrorSG.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSG.png
[Ennis-radianceSGLS.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSGLS.png
[Ennis-irradianceSGLS.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSGLS.png
[Ennis-irradianceErrorSGLS.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSGLS.png
[Ennis-radianceSGLSA.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSGLSA.png
[Ennis-irradianceSGLSA.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSGLSA.png
[Ennis-irradianceErrorSGLSA.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSGLSA.png
[Ennis-radianceSGNNLS.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/radianceSGNNLS.png
[Ennis-irradianceSGNNLS.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceSGNNLS.png
[Ennis-irradianceErrorSGNNLS.png]: https://github.com/kayru/Probulator/raw/master/Reports/Ennis/irradianceErrorSGNNLS.png
