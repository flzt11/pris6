from setuptools import setup, find_packages

setup(
    name='pris6',
    version='0.1',
    description='A package for performing various tasks related to dataset loading, evaluation, and visualization.',
    # long_description=open('README.md').read(),  # 可选，如果有 README 文件
    long_description_content_type='text/markdown',  # 如果使用 Markdown 格式
    author='Zhang Kailong',
    author_email='zhangkailong@bupt.edu.cn',
    url='https://github.com/flzt11/pris6.git',  # 如果有 GitHub 仓库链接
    packages=find_packages(),  # 自动发现所有包
    install_requires=[
        'scikit-learn==1.5',
        'matplotlib',
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # 根据你的环境选择合适的版本
)
