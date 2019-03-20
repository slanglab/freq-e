import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="freq-e",
    version="0.1.0",
    author="Katherine Keith and Brendan O'Connor",
    author_email="kkeith@cs.umass.edu",
    description="Class frequency estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slanglab/freq-e",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)