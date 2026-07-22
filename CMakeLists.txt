cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED MPITERATIVE_MAJOR)
	set(MPITERATIVE_MAJOR 0)
	set(MPITERATIVE_MINOR 1)
	set(MPITERATIVE_PATCH 0)
endif()

project(mp_iterative
	DESCRIPTION "Mixed-precision iterative methods: MTL5 linear algebra + Universal number systems"
	VERSION "${MPITERATIVE_MAJOR}.${MPITERATIVE_MINOR}.${MPITERATIVE_PATCH}"
	LANGUAGES CXX
	HOMEPAGE_URL "https://github.com/stillwater-sc/mp-iterative"
)

set(PROJECT_VERSION_MAJOR ${MPITERATIVE_MAJOR})
set(PROJECT_VERSION_MINOR ${MPITERATIVE_MINOR})
set(PROJECT_VERSION_PATCH ${MPITERATIVE_PATCH})

# Header-only interface library that composes MTL5 + Universal.
add_library(${PROJECT_NAME} INTERFACE)
add_library(sw::mp_iterative ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

target_include_directories(${PROJECT_NAME} INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

####
# Dependencies
#
# Strategy: try find_package first (for system/installed copies), then fall
# back to FetchContent from GitHub (both dependencies are header-only). Local
# checkouts can be used without network via, e.g.:
#   -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=/path/to/universal
#   -DFETCHCONTENT_SOURCE_DIR_MTL5=/path/to/mtl5
include(FetchContent)

# Universal number arithmetic library (header-only)
find_package(universal CONFIG QUIET)
if(universal_FOUND)
	target_link_libraries(${PROJECT_NAME} INTERFACE universal::universal)
	message(STATUS "Found Universal via find_package")
else()
	message(STATUS "Universal not found via find_package, fetching headers from GitHub")
	FetchContent_Declare(universal
		GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
		GIT_TAG        main
		GIT_SHALLOW    TRUE
	)
	# Header-only: populate but do NOT add_subdirectory (avoids building targets)
	FetchContent_GetProperties(universal)
	if(NOT universal_POPULATED)
		FetchContent_Populate(universal)
	endif()
	# Universal headers use two include conventions:
	#   - External: #include <sw/universal/...>  (needs include/)
	#   - Internal: #include <universal/...>     (needs include/sw/)
	target_include_directories(${PROJECT_NAME} INTERFACE
		$<BUILD_INTERFACE:${universal_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${universal_SOURCE_DIR}/include/sw>)
endif()

# MTL5 linear algebra library (header-only)
find_package(MTL5 CONFIG QUIET)
if(MTL5_FOUND)
	target_link_libraries(${PROJECT_NAME} INTERFACE MTL5::mtl5)
	message(STATUS "Found MTL5 via find_package")
else()
	message(STATUS "MTL5 not found via find_package, fetching headers from GitHub")
	FetchContent_Declare(mtl5
		GIT_REPOSITORY https://github.com/stillwater-sc/mtl5.git
		GIT_TAG        main
		GIT_SHALLOW    TRUE
	)
	# Header-only: populate but do NOT add_subdirectory (avoids building targets)
	FetchContent_GetProperties(mtl5)
	if(NOT mtl5_POPULATED)
		FetchContent_Populate(mtl5)
	endif()
	target_include_directories(${PROJECT_NAME} INTERFACE
		$<BUILD_INTERFACE:${mtl5_SOURCE_DIR}/include>)
endif()

####
# Compiler-specific settings
target_compile_options(${PROJECT_NAME} INTERFACE
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
	$<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
	$<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
	$<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Wpedantic>
)

####
# Build options
option(MPITERATIVE_BUILD_APPLICATIONS "Build demonstration applications" ON)
option(MPITERATIVE_BUILD_TESTS        "Build unit tests"                 ON)

if(MPITERATIVE_BUILD_APPLICATIONS)
	add_subdirectory(applications)
endif()

if(MPITERATIVE_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()

####
# Install
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME} EXPORT mp_iterativeTargets)

install(DIRECTORY include/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT mp_iterativeTargets
	FILE mp_iterativeTargets.cmake
	NAMESPACE sw::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mp_iterative
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/mp_iterativeConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/mp_iterativeConfig.cmake
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mp_iterative
)

write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/mp_iterativeConfigVersion.cmake
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/mp_iterativeConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/mp_iterativeConfigVersion.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mp_iterative
)
