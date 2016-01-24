add_definitions(-DNOMINMAX=1) # Never want min/max macros from Windows.h

add_executable(ProbulatorGUI
	Blitter.cpp
	Blitter.h
	Camera.cpp
	Camera.h
	Common.h
	Main.cpp
	Model.cpp
	Model.h
	Renderer.cpp
	Renderer.h
	Shaders.cpp
	Shaders.h
)
target_link_libraries(ProbulatorGUI 
	Probulator 
	gl3w 
	glfw 
	imgui
	nfd
	tinyobjloader
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	set(DATA_CMD COMMAND ${CMAKE_COMMAND} -E copy_directory) # CMake doesn't support symlinks on Windows
else()
	set(DATA_CMD COMMAND ${CMAKE_COMMAND} -E create_symlink)
endif()

if (MSVC)
	set(DATA_DIR $<TARGET_FILE_DIR:ProbulatorGUI>/../Data) # Default Visual Studio workind directory is $(ProjectDir)
else()
	set(DATA_DIR $<TARGET_FILE_DIR:ProbulatorGUI>/Data)
endif()

add_custom_command(TARGET ProbulatorGUI POST_BUILD 
	${DATA_CMD} ${CMAKE_SOURCE_DIR}/Data ${DATA_DIR})
