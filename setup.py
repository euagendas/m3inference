import setuptools

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

with open('LICENSE') as f:
    license = f.read()

setuptools.setup(
    name='m3inference',
    version='1.0',
    author='Zijian Wang et al.',
    author_email='zijwang@stanford.edu',
    description='M3 Inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    install_requires=reqs.strip().split('\n'),
    url='https://github.com/euagendas/m3inference',
    include_package_data=True,
    license=license,
    packages=setuptools.find_packages()

)