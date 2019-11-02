add_library(enkiTS STATIC 
	TaskScheduler.cpp
	TaskScheduler.h
	LockLessMultiReadPipe.h
)


if( UNIX )
	set( CMAKE_THREAD_PREFER_PTHREAD TRUE )
	find_package( Threads REQUIRED )
	if( CMAKE_USE_PTHREADS_INIT )
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread" )
	endif()
	target_link_libraries( enkiTS ${CMAKE_THREAD_LIBS_INIT} )
endif()

target_compile_features(enkiTS PRIVATE cxx_std_11)
target_include_directories(enkiTS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
