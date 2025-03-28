import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='MultiSetSR',
    version='0.0.1',
    author='Giorgio Morales - Montana State University',
    author_email='giorgiomoralesluna@gmail.com',
    description='Symbolic Regression using Transformers, genetic Algorithms, and genetic Programming',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NISL-MSU/MultiSetSR',
    project_urls={"Bug Tracker": "https://github.com/NISL-MSU/MultiSetSR/issues"},
    license='MIT',
    packages=setuptools.find_packages('src', exclude=['test']),
    # packages=setuptools.find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=['matplotlib', 'numpy', 'opencv-python', 'scipy', 'statsmodels', 'tqdm', 'timeout_decorator',
                      'h5py', 'pymoo==0.6.0', 'pyodbc', 'regex', 'tensorboard', 'python-dotenv', 'omegaconf', 'pandas'],
)
