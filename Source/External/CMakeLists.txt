# disable all warnings in external libs

if (MSVC)
	add_definitions(-W0)
else()
	add_definitions(-w)
endif()

# add external projects

add_subdirectory(gl3w)
add_subdirectory(glfw)
add_subdirectory(nfd)
add_subdirectory(stb)
add_subdirectory(tinyobjloader)
add_subdirectory(enkiTS)

include_directories(
	gl3w/include
	glfw/include
	glm
	enkiTS
)
add_subdirectory(imgui)

# put external projects into a dedicated IDE folder

set_property(TARGET gl3w PROPERTY FOLDER "External")
set_property(TARGET glfw PROPERTY FOLDER "External")
set_property(TARGET imgui PROPERTY FOLDER "External")
set_property(TARGET nfd PROPERTY FOLDER "External")
set_property(TARGET tinyobjloader PROPERTY FOLDER "External")
set_property(TARGET enkiTS PROPERTY FOLDER "External")
