import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tomopyui",
    version="0.0.1",
    description="GUI for tomopy built with ipywidgets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where='source'),
    package_dir={"": "source"},
    author='Sam Welborn',
    author_email='swelborn@slac.stanford.edu',
    keywords=['tomography', 'reconstruction', 'imaging'],
    platforms='Any',
    install_requires=[
        # 'tomopyui @ git+ssh://github.com/samwelborn/tomopyui@main',
         # Public repository
        # 'tomopy @ git+https://https://github.com/tomopy/tomopy@master',
    ],
    classifiers=[
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)