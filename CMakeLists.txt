cmake_minimum_required(VERSION 2.8)

project(Probulator)

if(MSVC)
	add_definitions(
		-D_CRT_SECURE_NO_WARNINGS
		-D_CRT_SECURE_NO_DEPRECATE
	)
endif()

add_subdirectory(Source)
