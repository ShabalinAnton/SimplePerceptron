cmake_minimum_required(VERSION 3.20)

project(simplePerceptron)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# You can also not to define it here but set CMAKE_TOOLCHAIN_FILE in command line
include(${CMAKE_BINARY_DIR}/generators/conan_toolchain.cmake)

find_package(OpenGL REQUIRED)
find_package(FreeGLUT REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS} ${freeglut_INCLUDE_DIRS_RELEASE})

add_executable(simplePerceptron simplePerceptron.cpp)
target_link_libraries(simplePerceptron ${OPENGL_LIBRARIES} FreeGLUT::freeglut_static)
