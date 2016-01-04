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
	add_library(nfd STATIC 
		nfd_common.c
		nfd_gtk.c
	)
endif()
