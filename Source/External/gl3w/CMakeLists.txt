cmake_minimum_required(VERSION 2.8)

project(gl3w)

add_library(gl3w src/gl3w.c)

if (UNIX)
  target_link_libraries(gl3w dl)
endif()

target_include_directories(gl3w PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)