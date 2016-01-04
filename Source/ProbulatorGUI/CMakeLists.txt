add_definitions(-DNOMINMAX=1) # Never want min/max macros from Windows.h

add_executable(ProbulatorGUI
	Blitter.cpp
	Blitter.h
	Common.h
	Main.cpp
	Model.cpp
	Model.h
	Renderer.cpp
	Renderer.h
)
target_link_libraries(ProbulatorGUI 
	Probulator 
	gl3w 
	glfw 
	imgui
	nfd
	tinyobjloader
)
