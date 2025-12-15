# setup.py - CORRECTED VERSION
from setuptools import setup, find_packages

setup(
    name="llm_essay_scoring",
    version="0.1.0",
    author="Sansiri Charoenpong",
    description="ECM3401 Individual Project",
    
    py_modules=['config'],  # Include loose .py files
    
    packages=find_packages(),  # Find package directories
    
    python_requires='>=3.10',
    install_requires=[
        'pandas>=2.0.3',
        'numpy>=1.24.3',
        'openai>=1.10.0',
        'transformers>=4.36.2',
        'sentence-transformers>=2.3.1',
        'faiss-cpu>=1.7.4',
        'tiktoken>=0.5.2',
        'tqdm>=4.66.1',
        'python-dotenv>=1.0.0',
    ],
)