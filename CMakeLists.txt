cmake_minimum_required(VERSION 3.20)
project(calibrate)
set(CMAKE_CXX_STANDARD 17)
add_executable(calibrate src/main.cpp src/calibrate.cpp)
target_include_directories(calibrate PRIVATE inc/)

find_package(OpenCV REQUIRED)
target_link_libraries(calibrate ${OpenCV_LIBS})
