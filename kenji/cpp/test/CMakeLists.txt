message(STATUS "Configure test")

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE include)

set(TestName test-cpp-deeplearning)

file(GLOB TEST_FILES src/*.cpp src/**/*.cpp src/**/**/*.cpp)
add_executable(${TestName} ${TEST_FILES})
target_link_libraries(${TestName} src_cpp Catch)

add_test(NAME UnitTests COMMAND ${TestName})
if (ENABLE_INTEGRATION_TEST)
    add_test(NAME IntegrationTests COMMAND sh integration_test.sh ${CMAKE_BINARY_DIR})
    add_test(NAME IntegrationScriptTests COMMAND sh integration_script_test.sh ${CMAKE_SOURCE_DIR})
    if (BUILD_TOOLS)
        add_test(NAME IntegrationToolsTests COMMAND sh integration_tools_test.sh ${CMAKE_BINARY_DIR})
    endif ()
endif ()

