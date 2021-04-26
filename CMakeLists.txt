cmake_minimum_required(VERSION 3.10)

# set the project name
project(FFT)

# add the executable
add_executable(FFT FFT.cpp)

find_package(PythonLibs REQUIRED)
INCLUDE_DIRECTORIES( 
    ${PYTHON_INCLUDE_DIRS} 
    "$ENV{USERPROFILE}/AppData/Roaming/Python/Python38/site-packages/numpy/core/include"
    )
link_directories( ${PYTHON_LIBRARIES})
target_link_libraries(FFT ${PYTHON_LIBRARY})