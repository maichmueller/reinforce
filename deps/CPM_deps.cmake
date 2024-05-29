# fetch and configure xtensor-python
if (ENABLE_BUILD_PYTHON_EXTENSION)
    CPMAddPackage(
            NAME xtensor-python  # The unique name of the dependency (should be the exported target's name)
            GIT_TAG 881b5a5           # The minimum version of the dependency (optional, defaults to 0)
            # Configuration options passed to the dependency (optional)
            SYSTEM TRUE
            DOWNLOAD_ONLY FALSE       # If set, the project is downloaded, but not configured (optional)
            GITHUB_REPOSITORY maichmueller/xtensor-python
    )
endif ()

# not yet needed
#CPMAddPackage(
#        NAME xtensor-blas  # The unique name of the dependency (should be the exported target's name)
#        GIT_TAG 0.20.0
#        # Configuration options passed to the dependency (optional)
#        SYSTEM TRUE
#        DOWNLOAD_ONLY FALSE       # If set, the project is downloaded, but not configured (optional)
#        GITHUB_REPOSITORY xtensor-stack/xtensor-blas
#)