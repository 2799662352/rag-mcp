from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="danbooru-bge-m3-rag",
    version="1.0.0",
    author="Danbooru RAG Team",
    author_email="contact@example.com",
    description="🚀 Professional RAG-MCP: Advanced Danbooru Prompt Generation System with BGE-M3 Triple Vector Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/2799662352/rag-mcp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "danbooru-rag=danbooru_prompt_server_v2_minimal:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    keywords="rag, embedding, danbooru, bge-m3, ai-art, prompt-generation, semantic-search",
    project_urls={
        "Bug Reports": "https://github.com/2799662352/rag-mcp/issues",
        "Source": "https://github.com/2799662352/rag-mcp",
        "Documentation": "https://github.com/2799662352/rag-mcp#readme",
    },
)