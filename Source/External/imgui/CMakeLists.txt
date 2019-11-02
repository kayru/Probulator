add_library(imgui STATIC 
	imgui.cpp
	imgui_demo.cpp
	imgui_draw.cpp
	imgui_impl_glfw_gl3.cpp
)
target_link_libraries(imgui glm gl3w glfw)
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
