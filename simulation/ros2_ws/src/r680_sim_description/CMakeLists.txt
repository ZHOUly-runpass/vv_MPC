cmake_minimum_required(VERSION 3.8)
project(r680_sim_description)

find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf meshes config maps DESTINATION share/${PROJECT_NAME})
ament_package()
