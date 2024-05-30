from setuptools import setup#, find_packages

__version__ = "1.0"

setup(
    name="ritnet",
    version=__version__,
    author="DeepInsight",
    #packages=find_packages(),
    author_email="hschoi@dinsight.ai",
    url="",
    description="RITnet",
    long_description="",
    zip_safe=False,
    force=True # force recompile the shared library. 
)