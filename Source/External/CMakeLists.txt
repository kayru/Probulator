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

set_property(TARGET gl3w PROPERTY FOLDER "External")
set_property(TARGET glfw PROPERTY FOLDER "External")
set_property(TARGET imgui PROPERTY FOLDER "External")
set_property(TARGET nfd PROPERTY FOLDER "External")
set_property(TARGET tinyobjloader PROPERTY FOLDER "External")
