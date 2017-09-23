message(STATUS "Target copy_data_dir")
add_custom_target(copy_data_dir ALL)
add_custom_command(TARGET copy_data_dir PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/data
        COMMENT "Copy the files in ${CMAKE_SOURCE_DIR}/data to ${CMAKE_BINARY_DIR}/data")
