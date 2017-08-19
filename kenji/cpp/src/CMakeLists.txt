message(STATUS "Configure src/code")
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/**/*.cpp)
list(REMOVE_ITEM SRC_FILES ${PROJECT_SOURCE_DIR}/src/cpp-deeplearning.cpp)
add_library(src STATIC ${SRC_FILES})

add_executable(cpp-deepleaning cpp-deepleaning.cpp)
target_link_libraries(cpp-deepleaning src)

if (BUILD_SHARED_LIBS)
    add_library(SRC_FILES SHARED ${SRC_FILES})
else ()
    add_library(SRC_FILES STATIC ${SRC_FILES})
endif ()
