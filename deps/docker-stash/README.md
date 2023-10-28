The purpose of this folder is to provide a transient cache for conan. 
This folder should be used as the `$CONAN_HOME` environment variable within any docker development images. 
ALL files and directories in this folder will be ignored by git! (except for this README) 

When defining your docker execution in CLion you can bind this folder in the toolchain settings e.g. via:

`-v /path/to/reinforce/deps/docker-stash/ubuntu22.04:/home/root/conan_cache`

where `path/to/reinforce/` is meant to be your system's path to the project folder of `reinforce`
and where it is assumed that the user of the image is `root` whose `$CONAN_HOME` variable maps to 
`/home/root/conan_cache`.

