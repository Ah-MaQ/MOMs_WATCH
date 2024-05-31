from setuptools import setup

__version__ = "1.0"

setup(
    name="l2cs",
    version=__version__,
    author="DeepInsight",
    #package_dir={"nia22": "py"},
    packages=['l2cs'],
    url="",
    description="NIA2022 L2CS Gaze tracking model",
    long_description="",
    zip_safe=False,
    force=True # force recompile the shared library. 
)
