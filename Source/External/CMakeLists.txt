add_subdirectory(gl3w)
add_subdirectory(glfw)

include_directories(
	gl3w/include
	glfw/include
)
add_subdirectory(imgui)
