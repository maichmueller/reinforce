function(set_project_compiler_options project_name)

    set(MSCV_COMPILER_OPTIONS
            "$<$<CXX_COMPILER_ID:MSVC>:"  # if Compiler == MSVC, add all of the following...
            # add /permissive- (language features that are not standard-compliant are errors)
            "/permissive-;"
            # add /EHsc, (compiler assumes that functions declared as extern "C" never throw a C++ exception)
            "/EHsc"
            # ncreases the number of sections that an object file can contain.
            "/bigobj"
            # compiler may create 1+ copies of itself, each in a separate process, that compile sources simultaneously.
            "/MP"
            # if build_type == debug, add -0d (NO optimizations and debug information)
            "$<$<CONFIG:DEBUG>:-Od>"
            # if build_type == release, add /O2 (max optimizations for msvc)
            "$<$<CONFIG:RELEASE>:/O2>"
            ">"
    )
    set(CLANG_COMPILER_OPTIONS
            "$<$<CXX_COMPILER_ID:Clang>:"  # if Compiler == Clang, add all of the following...
            # if build_type == debug, add -g3 (max debug information)
            "$<$<CONFIG:DEBUG>:-g3>"
            # if build_type == release, add -03 (max optimizations)
            "$<$<CONFIG:RELEASE>:-O3>"
            # if build_type == release and option:fast-math enabled, add -ffast-match
            "$<$<AND:$<CONFIG:RELEASE>,$<BOOL:${ENABLE_FAST_MATH}>>:-ffast-math>"
            # if platform == darwin (MAC), add -stdlib=libc++ (use the libc++ standard library, not gnu's stdlib)
            "$<$<PLATFORM_ID:Darwin>:-stdlib=libc++>"
            "-ftemplate-backtrace-limit=0"
            # various optimizations
            # (link time optimization, unrolling loops, omitting frame pointers(=increasing stack register mem for data))
            "$<$<CONFIG:RELEASE>:"
            "-flto"
            "-funroll-loops>"
            ">"
    )
    set(GCC_COMPILER_OPTIONS
            "$<$<CXX_COMPILER_ID:GNU>:"  # if Compiler == GCC, add all of the following...
            # if build_type == debug, add -g3 (max debug information)
            "$<$<CONFIG:DEBUG>:-g3>"
            # if build_type == release, add -03 (max optimizations)
            "$<$<CONFIG:RELEASE>:-O3>"
            # if build_type == release and option:fast-math enabled, add -ffast-match
            "$<$<AND:$<CONFIG:RELEASE>,$<BOOL:${ENABLE_FAST_MATH}>>:-ffast-math>"
            # the max level the error messages will traverse when running through concept constrained template code
            "-fconcepts-diagnostics-depth=10"
            "-ftemplate-backtrace-limit=0"
            # various optimizations
            # (link time optimization, unrolling loops, omitting frame pointers(=increasing stack register mem for data))
            "$<$<CONFIG:RELEASE>:"
            "-flto=auto"
            "-funroll-loops"
            "-ffat-lto-objects>"
            ">"
    )

    set(CLANG_LINK_OPTIONS
            "$<$<CXX_COMPILER_ID:Clang>:"  # if Compiler == Clang, add all of the following...
            "$<$<IN_LIST:-flto,${CLANG_COMPILER_OPTIONS}>:-fuse-ld=lld>"
            ">"
    )

    target_compile_options(
            ${project_name}
            INTERFACE
            ${MSCV_COMPILER_OPTIONS}
            ${CLANG_COMPILER_OPTIONS}
            ${GCC_COMPILER_OPTIONS}
    )

    target_link_options(
            ${project_name}
            INTERFACE
            ${CLANG_LINK_OPTIONS}
    )

endfunction()