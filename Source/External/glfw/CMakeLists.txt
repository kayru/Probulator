add_definitions(
	-D_GLFW_USE_OPENGL
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	add_definitions(
		-D_GLFW_WIN32
		-D_GLFW_WGL
		-D_GLFW_USE_DWM_SWAP_INTERVAL
	)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_definitions(
		-D_GLFW_COCOA
		-D_GLFW_NSGL
		-D_GLFW_USE_MENUBAR
	)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	add_definitions(
		-D_GLFW_X11
		-D_GLFW_GLX
		-D_GLFW_HAS_GLXGETPROCADDRESS
	)
endif()

set(glfw_src
	"src/context.c"
	"src/init.c"
	"src/input.c"
	"src/monitor.c"
	"src/window.c"
)

set(glfw_win32_src
	"src/wgl_context.c"
	"src/win32_init.c"
	"src/win32_monitor.c"
	"src/win32_time.c"
	"src/win32_tls.c"
	"src/win32_window.c"
	"src/winmm_joystick.c"
)

set(glfw_macosx_src
	"src/nsgl_context.m"
	"src/cocoa_init.m"
	"src/cocoa_monitor.m"
	"src/cocoa_window.m"
	"src/iokit_joystick.m"
	"src/mach_time.c"
	"src/posix_tls.c"
)

set(glfw_linux_src
	"src/glx_context.c"
	"src/x11_init.c"
	"src/x11_monitor.c"
	"src/x11_window.c"
	"src/xkb_unicode.c"
	"src/linux_joystick.c"
	"src/posix_time.c"
	"src/posix_tls.c"
)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	set(glfw_src ${glfw_src} ${glfw_win32_src})
endif()

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(glfw_src ${glfw_src} ${glfw_linux_src})
endif()

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(glfw_src ${glfw_src} ${glfw_macosx_src})
endif()

add_library(glfw STATIC ${glfw_src})

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	target_link_libraries(glfw
		opengl32
	)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	find_package(OpenGL)
	target_link_libraries(glfw
		${OPENGL_LIBRARY}
		pthread
		X11
		Xxf86vm
		Xrandr
		Xi
	)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	find_library(appkit_lib AppKit)
	find_library(corevideo_lib CoreVideo)
	find_library(foundation_lib Foundation)
	find_library(iokit_lib IOKit)
	find_library(opengl_lib OpenGL)
	target_link_libraries(glfw
		${appkit_lib}
		${corevideo_lib}
		${foundation_lib}
		${iokit_lib}
		${opengl_lib}
		${EMBREE_LIBRARY}
	)
endif()
