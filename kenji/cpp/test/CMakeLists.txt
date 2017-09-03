message(STATUS "Configure test")

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE include)

set(TestName test-cpp-deeplearning)

file(GLOB_RECURSE TEST_FILES "*.cpp")
add_executable(${TestName} ${TEST_FILES})
target_link_libraries(${TestName} src_cpp Catch)

add_test(NAME UnitTests COMMAND ${TestName})
if (ENABLE_INTEGRATION_TEST)
    add_test(NAME IntegrationTests COMMAND sh integration_test.sh ${CMAKE_BINARY_DIR})
endif ()

