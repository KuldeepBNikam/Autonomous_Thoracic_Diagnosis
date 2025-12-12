from setuptools import find_packages, setup
from typing import List

# Constant for "-e ."
HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Reads requirements.txt and returns a clean list of dependencies.
    Removes '-e .' if present.
    """
    requirements = []
    
    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]

        # Remove editable install flag if present
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(

    name = "HyperLung-XR",
    version = "0.0.1",
    author = "Kuldeep Nikam",
    author_email= "kuldeepnikam2782@gmail.com",
    install_requires = get_requirements(),
    package = find_packages()
)