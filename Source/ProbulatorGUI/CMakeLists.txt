add_executable(ProbulatorGUI
	Common.h
	Main.cpp
	Blitter.h
	Blitter.cpp
	Renderer.h
	Renderer.cpp
)
target_link_libraries(ProbulatorGUI Probulator gl3w glfw)
