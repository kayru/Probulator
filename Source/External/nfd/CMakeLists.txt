include_directories(include)


if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	add_library(nfd STATIC 
		nfd_common.c
		nfd_win.cpp
	)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_library(nfd STATIC 
		nfd_common.c
		nfd_cocoa.m
	)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

	find_package(PkgConfig REQUIRED)
	pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

	include_directories(${GTK3_INCLUDE_DIRS})
	link_directories(${GTK3_LIBRARY_DIRS})
	add_definitions(${GTK3_CFLAGS_OTHER})

	add_library(nfd STATIC 
		nfd_common.c
		nfd_gtk.c
	)

	target_link_libraries(nfd ${GTK3_LIBRARIES})
endif()
