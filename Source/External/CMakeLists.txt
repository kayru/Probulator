add_subdirectory(gl3w)
add_subdirectory(glfw)
add_subdirectory(tinyobjloader)
add_subdirectory(nfd)

include_directories(
	gl3w/include
	glfw/include
	glm
)
add_subdirectory(imgui)
