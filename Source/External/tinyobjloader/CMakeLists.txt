if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-std=c++11)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
	add_definitions(-std=c++11 -stdlib=libc++)
endif()

add_library(tinyobjloader STATIC tiny_obj_loader.cc)
