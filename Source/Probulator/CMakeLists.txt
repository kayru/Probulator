if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-std=c++11)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	add_definitions(-std=c++11 -stdlib=libc++)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	find_path(tbbinclude tbb)
	include_directories(${tbbinclude})
endif()

add_executable(Probulator
	Common.h
	Image.h
	Math.h
	SphericalGaussian.h
	SphericalHarmonics.h
	Thread.h
	Image.cpp
	Main.cpp
	SphericalGaussian.cpp
	SphericalHarmonics.cpp
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	find_library(tbb tbb)
	target_link_libraries(Probulator ${tbb})
endif()
