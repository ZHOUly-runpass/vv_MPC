from glob import glob
from setuptools import find_packages, setup

package_name = "r680_sim_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="R680 MPC Project",
    maintainer_email="maintainer@example.com",
    description="R680 Gazebo Classic simulation bringup",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "pointcloud_field_adapter = r680_sim_bringup.pointcloud_field_adapter:main",
        "ground_truth_bridge = r680_sim_bringup.ground_truth_bridge:main",
        "dynamic_obstacle_controller = r680_sim_bringup.dynamic_obstacle_controller:main",
        "benchmark_manager = r680_sim_bringup.benchmark_manager:main",
        "route_publisher = r680_sim_bringup.route_publisher:main",
        "runtime_audit = r680_sim_bringup.runtime_audit:main",
    ]},
)
