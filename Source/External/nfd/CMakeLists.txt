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

	find_package(PkgConfig REQUIRED)
	pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

	link_directories(${GTK3_LIBRARY_DIRS})
	target_include_directories(nfd PRIVATE ${GTK3_INCLUDE_DIRS})
	target_link_libraries(nfd ${GTK3_LIBRARIES})
	target_compile_options(nfd PRIVATE ${GTK3_CFLAGS_OTHER})

endif()

target_include_directories(nfd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
