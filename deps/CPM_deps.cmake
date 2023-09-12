# fetch and configure xtensor-python
CPMAddPackage(
        NAME xtensor-python  # The unique name of the dependency (should be the exported target's name)
        GIT_TAG 0.26.1            # The minimum version of the dependency (optional, defaults to 0)
        # Configuration options passed to the dependency (optional)
        SYSTEM TRUE
        DOWNLOAD_ONLY FALSE       # If set, the project is downloaded, but not configured (optional)
        GITHUB_REPOSITORY xtensor-stack/xtensor-python
)
add_definitions(-DHAVE_CBLAS=1)
if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()
message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
CPMAddPackage(
        NAME xtensor-blas  # The unique name of the dependency (should be the exported target's name)
        GIT_TAG 0.20.0
        # Configuration options passed to the dependency (optional)
        SYSTEM TRUE
        DOWNLOAD_ONLY FALSE       # If set, the project is downloaded, but not configured (optional)
        GITHUB_REPOSITORY xtensor-stack/xtensor-blas
)