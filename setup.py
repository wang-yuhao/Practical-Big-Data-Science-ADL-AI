import os
from setuptools import find_packages, setup

install_requires = []
with open('requirements.txt', 'r+') as f:
    for line in f:
        if '-e ' in line:
            dependency_name = line.split('=')[1].strip()
            line = dependency_name + ' @ ' + line[2:].strip()
        
        if line == "" or line.startswith('#'):
            continue

        install_requires.append(line.strip())

tests_require = []
with open('requirements-dev.txt', 'r+') as f:
    for line in f:
        if '-e' in line:
            dependency_name = line.split('=')[1].strip()
            line = dependency_name + ' @ ' + line[2:].strip()
        
        if line == "" or line.startswith('#'):
            continue

        tests_require.append(line.strip())


setup(
    name='adlai-kge',
    packages=find_packages(),
    version='0.0.1',
    description='Link Prediction for Knowledge Graphs',
    author='adl-ai',
    license='MIT',
    install_requires=install_requires,
    test_require=tests_require,
    scripts=list(map(lambda f: os.path.join('scripts/', f), os.listdir('scripts/')))
)
