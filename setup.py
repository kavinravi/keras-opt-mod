"""Package configuration."""
import setuptools

setuptools.setup(
    name='keras-opt',
    author='Pedro Marques',
    author_email='pedro.r.marques@gmail.com',
    description='Scipy keras optimizer',
    url='https://github.com/kavinravi/keras-opt-mod',
    packages=setuptools.find_packages(include=['keras_opt', 'keras_opt.*']),
    install_requires=['numpy>=1.23', 'tensorflow>=2.12', 'scipy>=1.10'],
    extras_require={'test': ['pytest>=7', 'ruff', 'scikit-learn']},
    python_requires='>=3.9',
    version='0.0.8',
)
