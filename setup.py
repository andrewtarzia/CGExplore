import setuptools

setuptools.setup(
    name="CGExplore",
    version="0.0.1",
    author="Andrew Tarzia",
    author_email="andrew.tarzia@gmail.com",
    description="Library for Minimalistic model optimisation.",
    url="https://github.com/andrewtarzia/CGExplore",
    packages=setuptools.find_packages(),
    install_requires=(
        'numpy',
        'scipy',
        'sklearn',
        'stk',
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
