from setuptools import setup, find_packages

setup(
    name='cvxstoc',
    version='0.1.0',
    description='Reimplementation of cvxstoc using numpyro, jax, and cvxpy',
    author='Joseph Sakaya',
    author_email='hosanna.joseph@gmail.com',
    url='https://github.com/jsakaya/cvxstoc',  
    packages=find_packages(),
    install_requires=[
        'numpy',
        'numpyro',
        'jax',
        'cvxpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.10',
)

