from setuptools import find_packages, setup

package_name = "zss_collection"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    package_data={"": ["py.typed"]},
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Abinav2695",
    maintainer_email="abinav2695@gmail.com",
    description="Modular data collection pipeline for AIC insertion task training.",
    license="TODO: License declaration",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "collection_node      = zss_collection.node:main",
            "collection_node_once = zss_collection.node_once:main",
        ],
    },
)
