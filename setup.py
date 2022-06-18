from setuptools import setup, find_packages


with open("README.md", "r") as readme:
    long_description = readme.read()

with open('structured_data_profiling/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="stuctured-profiling",
    version="0.0.2",
    author="Clearbox AI",
    author_email="info@clearbox.ai",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clearbox-AI/",
    install_requires=requirements,
    packages=find_packages(),
    python_requires='>=3.6.2',
)