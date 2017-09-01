include(CMakeToolsHelpers OPTIONAL) # for Visual Studio Code
cmake_minimum_required(VERSION 3.3.0 FATAL_ERROR)

# Make PROJECT_SOURCE_DIR, PROJECT_BINARY_DIR, and PROJECT_NAME available.
project(cpp-deep-learning CXX)

#------------------------------------------------------
# Build type
#------------------------------------------------------

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif ()

OPTION(BUILD_SHARED_LIBS "Set to OFF to build static libraries" ON)
OPTION(USE_OPENMP "Enable/Disable OpenMP" ON)
OPTION(BUILD_TESTS "Set to OFF to not compile tests" ON)

OPTION(DISABLE_OPTIMIZATION_IN_RELEASE "Disable optimization in release" OFF)

#-------------------
# set common include folder for module
#-------------------
set(HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

include_directories(${HEADER_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src/*.h)
# set(CMAKE_CXX_STANDARD 11)
# add_definitions("-std=c++14")   # Adds -D define flags to the compilation of source files.
# add_compile_options(-std=c++14) # Adds options to the compilation of source files.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------------------------------------------------------
#   Find Dependencies
# ----------------------------------------------------------------------------

find_package(PkgConfig REQUIRED)

# OpenMP
if (USE_OPENMP)
    if (APPLE)
        find_program(CLANG_OMP_PP ${CMAKE_CXX_COMPILER})
        if (CLANG_OMP_PP)
            message(STATUS "Found ${CMAKE_CXX_COMPILER} in ${CLANG_OMP_PP}")
        endif ()
    endif ()
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(LINKER_LIBS ${LINKER_LIBS} gomp)
        message(STATUS "Enabled OpenMP")
    endif ()
endif ()

# Eigen
pkg_search_module(Eigen3 REQUIRED eigen3)

# ----------------------------------------------------------------------------
#   Uninstall target, for "make uninstall"
# ----------------------------------------------------------------------------
CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

# ----------------------------------------------------------------------------
#   Program Optimization and debug (Extracted from OpenCV)
# ----------------------------------------------------------------------------
set(WARNINGS_ARE_ERRORS OFF CACHE BOOL "Treat warnings as errors")
set(WHOLE_PROGRAM_OPTIMIZATION OFF CACHE BOOL "Flags for whole program optimization.")

set(EXTRA_C_FLAGS "")
set(EXTRA_C_FLAGS_RELEASE "")
set(EXTRA_C_FLAGS_DEBUG "")
set(EXTRA_EXE_LINKER_FLAGS "")
set(EXTRA_EXE_LINKER_FLAGS_RELEASE "")
set(EXTRA_EXE_LINKER_FLAGS_DEBUG "")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
IF (CMAKE_COMPILER_IS_GNUCXX OR MINGW OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")
    set(USE_OMIT_FRAME_POINTER ON CACHE BOOL "Enable -fomit-frame-pointer for GCC")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES arm*) # We can use only -O2 because the -O3 causes gcc crash
        set(USE_O2 ON CACHE BOOL "Enable -O2 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
    endif ()
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES powerpc*)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(USE_POWERPC ON CACHE BOOL "Enable PowerPC for GCC")
    endif ()
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64* OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
        set(USE_MMX ON CACHE BOOL "Enable MMX for GCC")
        set(USE_SSE ON CACHE BOOL "Enable SSE for GCC")
        set(USE_SSE2 ON CACHE BOOL "Enable SSE2 for GCC")
        set(USE_SSE3 ON CACHE BOOL "Enable SSE3 for GCC")
    endif ()
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES i686* OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES x86)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
        set(USE_MMX ON CACHE BOOL "Enable MMX for GCC")
        set(USE_SSE OFF CACHE BOOL "Enable SSE for GCC")
        set(USE_SSE2 OFF CACHE BOOL "Enable SSE2 for GCC")
        set(USE_SSE3 OFF CACHE BOOL "Enable SSE3 for GCC")
    endif ()

    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall")

    if (WARNINGS_ARE_ERRORS)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Werror")
    endif ()

    # The -Wno-long-long is required in 64bit systems when including sytem headers.
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64* OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    endif ()

    # Whole program optimization
    if (WHOLE_PROGRAM_OPTIMIZATION)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fwhole-program --combine")
    endif ()

    # Other optimizations
    if (USE_OMIT_FRAME_POINTER)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fomit-frame-pointer")
    endif ()
    if (USE_O2)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O2")
    endif ()
    if (USE_O3)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    endif ()
    if (USE_FAST_MATH)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -ffast-math")
    endif ()
    if (USE_POWERPC)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
    endif ()
    if (USE_MMX)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mmmx")
    endif ()
    if (USE_SSE)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse")
    endif ()
    if (USE_SSE2)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse2")
    endif ()
    if (USE_SSE3 AND NOT MINGW) # SSE3 should be disabled under MingW because it generates compiler errors
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse3")
    endif ()

    if (ENABLE_PROFILING)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -pg -g")
    else ()
        if (NOT APPLE)
            set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
        endif ()
    endif ()

    # Parallel mode
    IF (${CMAKE_SYSTEM_PROCESSOR} MATCHES armv7l) # In ARM_COrtex8 with neon, enalble vectorized operations
        SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mcpu=cortex-a8 -mfpu=neon -mfloat-abi=softfp -ftree-vectorize ")
    ENDIF ()


    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -DNDEBUG")
    set(EXTRA_C_FLAGS_DEBUG "-g3 -O0 -DDEBUG -D_DEBUG -W -Wextra -Wno-return-type")

    MESSAGE(STATUS "-------------------------------------------------------------------------------")
    message(STATUS "GNU COMPILER")
    MESSAGE(STATUS "-------------------------------------------------------------------------------")

    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS} -std=c++14")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
    if (DISABLE_OPTIMIZATION_IN_RELEASE)
        SET(CMAKE_CXX_FLAGS_RELEASE "-O0 -DNDEBUG")
    else ()
        SET(CMAKE_CXX_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}")
    endif ()
    SET(CMAKE_CXX_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE}")
    SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG}")
ENDIF ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-inconsistent-missing-override")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment")
endif ()

#------------------------------------------------
# DIRS
#------------------------------------------------
add_subdirectory(src)
# add_subdirectory(script)

# ----------------------------------------------------------------------------
#   Testing
# ----------------------------------------------------------------------------
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

# ----------------------------------------------------------------------------
# display status message for important variables
# ----------------------------------------------------------------------------
message(STATUS)
MESSAGE(STATUS "-------------------------------------------------------------------------------")
message(STATUS "General configuration for ${PROJECT_NAME} ${PROJECT_VERSION}")
MESSAGE(STATUS "-------------------------------------------------------------------------------")
message(STATUS)
message("    Compiler:" "${CMAKE_COMPILER}" "${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_BINARY_DIR:         ${CMAKE_BINARY_DIR}")
MESSAGE(STATUS)
MESSAGE(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
MESSAGE(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
MESSAGE(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
MESSAGE(STATUS)
MESSAGE(STATUS "---------------------------     Documentation     -----------------------------")
MESSAGE(STATUS)
MESSAGE(STATUS "PCL_LIBRARY_DIRS=${PCL_LIBRARY_DIRS}") # PCL_INCLUDE_DIRS=${PCL_INCLUDE_DIRS}")
MESSAGE(STATUS "OpenCV_LIB_DIR=${OpenCV_LIBS} OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LIBUSB1_INCLUDE_DIRS=${LIBUSB1_INCLUDE_DIRS}")
MESSAGE(STATUS "LIB_REALSENSE=${LIB_REALSENSE}")
MESSAGE(STATUS "LIB_ARUCO=${LIB_ARUCO}")
MESSAGE(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}")
MESSAGE(STATUS "USE_OWN_EIGEN3=${USE_OWN_EIGEN3} EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")
MESSAGE(STATUS)
MESSAGE(STATUS "---------------------------     Options     -----------------------------")
MESSAGE(STATUS)
MESSAGE(STATUS "BUILD_SHARED_LIBS:               ${BUILD_SHARED_LIBS}")
MESSAGE(STATUS "USE_OPENMP:                      ${USE_OPENMP}")
MESSAGE(STATUS "BUILD_TESTS:                     ${BUILD_TESTS}")
MESSAGE(STATUS)
MESSAGE(STATUS "Change a value with: cmake -D<Variable>=<Value>")
MESSAGE(STATUS)
