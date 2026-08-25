cmake_minimum_required(VERSION 3.8)
project(r680_sim_worlds)

find_package(ament_cmake REQUIRED)
install(DIRECTORY worlds models DESTINATION share/${PROJECT_NAME})
ament_package()
