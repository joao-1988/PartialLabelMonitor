from setuptools import setup, find_packages

setup(
    name="PartialLabelMonitor",
    version="0.0.0",
    packages=find_packages(),

    # Metadata
    author="João Flávio Andrade Silva",
    author_email="joaoflavio1988@gmail.com",
    description="Monitoring models with partially labeled data.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/joao-1988/PartialLabelMonitor",
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License"
    ],

    # Optional
    install_requires=['pandas']
)