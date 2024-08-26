from setuptools import setup, find_packages
setup(
    name='antgen',
    version='0.4.1',
    author='Nuno Velosa',
    author_email='nunovelosa@hotmail.com',
    description='This tool generates synthetic macroscopic load signatures for their use in conjunction with NILM (load disaggregation) tools.',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    # Non-python files to include in pypi
    package_data={'': ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '*/*/*/*/*/*']},
)
