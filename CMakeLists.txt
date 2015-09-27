cmake_minimum_required(VERSION 2.8)

project(Probulator)

if(MSVC)
	add_definitions(
		-D_CRT_SECURE_NO_WARNINGS
		-D_CRT_SECURE_NO_DEPRECATE
	)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /Zo")
endif()

add_subdirectory(Source)
