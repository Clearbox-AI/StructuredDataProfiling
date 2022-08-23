from setuptools import setup, find_packages


with open("README.md") as readme:
    long_description = readme.read()

requirements = [
    "dateparser>=1.0.0",
    "distfit>=1.4.0",
    "great-expectations>=0.13.41",
    "joblib>=1.1.0",
    "numpy>=1.20",
    "pandas>=1.3.0",
    "pytest>=5.4.3",
    "pytest-lazy-fixture>=0.6.3",
    "scikit-learn>=1.0",
    "scipy>=1.7.2",
    "twine>=4.0.1",
]

setup(
    name="structured-profiling",
    version="0.1.4.1",
    author="Clearbox AI",
    author_email="info@clearbox.ai",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clearbox-AI/",
    install_requires=requirements,
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6.2",
    include_package_data=True,
)
