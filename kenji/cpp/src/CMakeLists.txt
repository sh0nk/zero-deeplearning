message(STATUS "Configure src/code")
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/**/*.cpp)
list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/cpp-deeplearning.cpp)
add_library(src STATIC ${SRC_FILES})

if (BUILD_SHARED_LIBS)
  add_library(src_cpp SHARED ${SRC_FILES})
else ()
  add_library(src_cpp STATIC ${SRC_FILES})
endif ()

add_executable(cpp-deeplearning cpp-deeplearning.cpp)
target_link_libraries(cpp-deeplearning src_cpp ${PYTHON_LIBRARIES})
