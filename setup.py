import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easy_dnn", # Replace with your own username
    version="0.0.1",
    author="Yishai Rasowsky",
    author_email="yishairasowsky@gmail.com",
    description="An easy way to show how a simple deep neural net outperforms a logistic regression classifier.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yishairasowsky/nn_vs_log_reg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
