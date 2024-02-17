# [Choice] bionic (18.04), focal (20.04), jammy (22.04)
ARG VARIANT="jammy"
ARG USERNAME="root"
FROM ubuntu:${VARIANT}
# Restate the build args to use it in the coming sections (otherwise they are missing)
ARG VARIANT
ARG USERNAME

SHELL ["/bin/bash", "-c"]

# Install necessary packages available from standard repos
RUN apt-get update -qq && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        software-properties-common wget apt-utils file zip \
        openssh-client gpg-agent socat rsync \
        make ninja-build git \
        python3 python3-pip python3-dev \
        liblapack-dev  # required for tensor libraries and can't be handled by conan (yet)

# Install conan
RUN python3 -m pip install --upgrade pip setuptools && \
    python3 -m pip install numpy conan && \
    conan --version

# By default, anything you run in Docker is done as superuser.
# Conan runs some install commands as superuser, and will prepend `sudo` to
# these commands, unless `CONAN_SYSREQUIRES_SUDO=0` is in your env variables.
ENV CONAN_SYSREQUIRES_SUDO 0
# Some packages request that Conan use the system package manager to install
# a few dependencies. This flag allows Conan to proceed with these installations;
# leaving this flag undefined can cause some installation failures.
ENV CONAN_SYSREQUIRES_MODE enabled

ENV CONAN_HOME /home/${USERNAME}/conan_cache

# User-settable versions:
# This Dockerfile should support gcc-[7, 8, 9, 10, 11, 12, 13] and clang-[10, 11, 12, 13, 14, 15, 16, 17]
# Earlier versions of clang will require significant modifications to the IWYU section
ARG GCC_VER="13"
# Add gcc-${GCC_VER}
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update -qq && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        gcc-${GCC_VER} g++-${GCC_VER} gdb

# Set gcc-${GCC_VER} as default gcc
RUN update-alternatives --install /usr/bin/gcc gcc $(which gcc-${GCC_VER}) 100
RUN update-alternatives --install /usr/bin/g++ g++ $(which g++-${GCC_VER}) 100

ARG LLVM_VER="17"
# Add clang-${LLVM_VER}
ARG LLVM_URL="http://apt.llvm.org/${VARIANT}/"
ARG LLVM_PKG="llvm-toolchain-${VARIANT}-${LLVM_VER}"
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - 2>/dev/null && \
    add-apt-repository -y "deb ${LLVM_URL} ${LLVM_PKG} main" && \
    apt-get update -qq && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        clang-${LLVM_VER} lldb-${LLVM_VER} lld-${LLVM_VER} clangd-${LLVM_VER} \
        llvm-${LLVM_VER}-dev libclang-${LLVM_VER}-dev clang-tidy-${LLVM_VER}

# Set the default clang-tidy, so CMake can find it
RUN update-alternatives --install /usr/bin/clang-tidy clang-tidy $(which clang-tidy-${LLVM_VER}) 1

# Set clang-${LLVM_VER} as default clang
RUN update-alternatives --install /usr/bin/clang clang $(which clang-${LLVM_VER}) 100
RUN update-alternatives --install /usr/bin/clang++ clang++ $(which clang++-${LLVM_VER}) 100

# Add current cmake/ccmake, from Kitware
ARG CMAKE_URL="https://apt.kitware.com/ubuntu/"
ARG CMAKE_PKG=${VARIANT}
ARG CMAKE_VER=3.26.4
RUN export DEBIAN_FRONTEND=noninteractive; \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository -y "deb ${CMAKE_URL} ${CMAKE_PKG} main"

RUN export DEBIAN_FRONTEND=noninteractive; \
#    apt-get update -qq && \
    apt list -a cmake | grep "${CMAKE_VER}"
RUN export DEBIAN_FRONTEND=noninteractive; \
    export CMAKE_VERSION_STRING=$(apt list -a cmake 2>/dev/null | grep ${CMAKE_VER} | cut -d" " -f2); \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    cmake=${CMAKE_VERSION_STRING} \
    cmake-data=${CMAKE_VERSION_STRING} \
    cmake-curses-gui=${CMAKE_VERSION_STRING}


# Install editors
RUN apt-get update -qq && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        neovim emacs nano

# Install optional dependecies
RUN apt-get update -qq && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        doxygen graphviz ccache cppcheck ssh

# Install include-what-you-use
ENV IWYU /home/iwyu
ENV IWYU_BUILD ${IWYU}/build
ENV IWYU_SRC ${IWYU}/include-what-you-use
RUN mkdir -p ${IWYU_BUILD} && \
    git clone --branch clang_${LLVM_VER} \
        https://github.com/include-what-you-use/include-what-you-use.git \
        ${IWYU_SRC}
RUN CC=clang-${LLVM_VER} CXX=clang++-${LLVM_VER} cmake -S ${IWYU_SRC} \
        -B ${IWYU_BUILD} \
        -G "Unix Makefiles" -DCMAKE_PREFIX_PATH=/usr/lib/llvm-${LLVM_VER} && \
    cmake --build ${IWYU_BUILD} -j && \
    cmake --install ${IWYU_BUILD}

# Per https://github.com/include-what-you-use/include-what-you-use#how-to-install:
# `You need to copy the Clang include directory to the expected location before
#  running (similarly, use include-what-you-use -print-resource-dir to learn
#  exactly where IWYU wants the headers).`
RUN mkdir -p $(include-what-you-use -print-resource-dir 2>/dev/null)
RUN ln -s $(readlink -f /usr/lib/clang/${LLVM_VER}/include) \
    $(include-what-you-use -print-resource-dir 2>/dev/null)/include

## Cleanup cached apt data we don't need anymore
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Allow the user to set compiler defaults
ARG USE_CLANG
# if --build-arg USE_CLANG=1, set CC to 'clang' or set to null otherwise.
ENV CC=${USE_CLANG:+"clang"}
ENV CXX=${USE_CLANG:+"clang++"}
# if CC is null, set it to 'gcc' (or leave as is otherwise).
ENV CC=${CC:-"gcc"}
ENV CXX=${CXX:-"g++"}

RUN echo $CC && echo $CXX

# create the default profile for conan to rely on
RUN conan profile detect
RUN echo $' \n\
include(default) \n\
[settings]  \n\
compiler.cppstd=17  \n\
os=Linux \n\
\n\
[conf] \n\
tools.build:jobs=8 \
    ' >> /home/conan_profile

RUN conan config install /home/conan_profile

CMD ["/bin/bash"]