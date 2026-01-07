# setup.py
# ECM3401: Measuring Semantic Robustness in LLM-Based Essay Scoring
# Installation: pip install -e .  (development mode)

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    # =========================================================================
    # Package Metadata
    # =========================================================================
    name="llm_essay_scoring",
    version="1.0.0",
    author="Sansiri Charoenpong",
    author_email="sc1076@exeter.ac.uk",
    description="Measuring Semantic Robustness in LLM-Based Essay Scoring (ECM3401 Individual Project)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ECM3401-LLM-Essay-Scoring",  # Update if you have a repo
    
    # =========================================================================
    # Package Structure
    # =========================================================================
    # Include the config.py module (loose file in project root)
    py_modules=['config'],
    
    # Find all package directories (if any)
    packages=find_packages(),
    
    # Include non-Python files specified in MANIFEST.in
    include_package_data=True,
    
    # =========================================================================
    # Python Version Requirements
    # =========================================================================
    python_requires='>=3.10',
    
    # =========================================================================
    # Core Dependencies
    # Minimal set required for the package to run
    # Full dependencies in requirements.txt
    # =========================================================================
    install_requires=[
        'pandas>=2.0.3',
        'numpy>=1.24.4',
        'scipy>=1.11.4',
        'scikit-learn>=1.3.2',
        'torch>=2.1.2',
        'openai>=2.11.0',
        'transformers>=4.44.2',
        'accelerate>=0.34.2',
        'sentence-transformers>=2.7.0',
        'faiss-cpu>=1.8.0',
        'tiktoken>=0.7.0',
        'nltk>=3.8.1',
        'textstat>=0.7.3',
        'matplotlib>=3.8.4',
        'seaborn>=0.13.2',
        'tqdm>=4.66.5',
        'python-dotenv>=1.0.1',
        'pyyaml>=6.0.2',
        'requests>=2.32.3',
        'pydantic>=2.9.2',
    ],
    
    # =========================================================================
    # Optional Dependencies
    # Install with: pip install -e ".[dev]" or pip install -e ".[jupyter]"
    # =========================================================================
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'jupyterlab>=4.0.13',
            'ipykernel>=6.29.5',
        ],
    },
    
    # =========================================================================
    # Package Classification
    # =========================================================================
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: MacOS :: MacOS X',
        'License :: OSI Approved :: MIT License',
    ],
    
    # =========================================================================
    # Keywords for Discovery
    # =========================================================================
    keywords='llm essay scoring cefr robustness nlp education assessment',
    
    # =========================================================================
    # Project URLs
    # =========================================================================
    project_urls={
        'Documentation': 'https://github.com/yourusername/ECM3401-LLM-Essay-Scoring/blob/main/README.md',
        'Source': 'https://github.com/yourusername/ECM3401-LLM-Essay-Scoring',
        'Supervisor': 'https://www.exeter.ac.uk/staff/profile/index.php?web_id=Rodrigo_Wilkens',
    },
)

# ============================================================================
# INSTALLATION NOTES
# ============================================================================
# Development Installation (recommended):
#   pip install -e .
#   This installs the package in "editable" mode, so changes to code are
#   immediately reflected without reinstalling.
#
# Regular Installation:
#   pip install .
#   This installs the package normally.
#
# With Optional Dependencies:
#   pip install -e ".[dev]"      # Development tools
#   pip install -e ".[jupyter]"  # Jupyter support
#   pip install -e ".[dev,jupyter]"  # Both
#
# Verification:
#   python -c "import config; print('Package installed successfully!')"
#   python -c "from config import CORPUS_FILE; print(f'Corpus: {CORPUS_FILE}')"
#
# What This Does:
#   1. Makes 'config' importable from anywhere in the project
#   2. Installs all dependencies listed in install_requires
#   3. Sets up the project as a Python package
#   4. Allows you to run scripts using: python scripts/script_name.py
#      OR using module syntax: python -m scripts.script_name
#
# When to Use:
#   - Run this once after cloning the project
#   - Re-run if you modify setup.py or add new dependencies
#   - Use "pip install -e ." for active development
#   - Use "pip install ." for final installation
#
# Benefits of Using setup.py:
#   ✓ Clean imports: "from config import DATASET_ROOT" works everywhere
#   ✓ Package management: pip tracks installed packages
#   ✓ Dependency resolution: automatically installs required packages
#   ✓ Editable mode: changes reflected immediately
#   ✓ Distribution: can share package with others easily