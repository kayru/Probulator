# disable all warnings in external libs

if (MSVC)
	add_definitions(-W0)
else()
	add_definitions(-w)
endif()

# add external projects

add_subdirectory(eigen)
add_subdirectory(enkiTS)
add_subdirectory(gl3w)
add_subdirectory(glfw)
add_subdirectory(glm)
add_subdirectory(imgui)
add_subdirectory(nfd)
add_subdirectory(stb)
add_subdirectory(tinyobjloader)

# put external projects into a dedicated IDE folder

#set_property(TARGET eigen PROPERTY FOLDER "External")
set_property(TARGET enkiTS PROPERTY FOLDER "External")
set_property(TARGET gl3w PROPERTY FOLDER "External")
set_property(TARGET glfw PROPERTY FOLDER "External")
#set_property(TARGET glm PROPERTY FOLDER "External")
set_property(TARGET imgui PROPERTY FOLDER "External")
set_property(TARGET nfd PROPERTY FOLDER "External")
set_property(TARGET stb PROPERTY FOLDER "External")
set_property(TARGET tinyobjloader PROPERTY FOLDER "External")
