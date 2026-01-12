from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="orchestra-llm-cache",
    version="0.1.0",
    author="Orchestra Team",
    author_email="team@orchestra.ai",
    description="Reduce AI orchestration costs by 85% with semantic caching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uejsh/orchestra",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
    ],
    extras_require={
        "langgraph": ["langgraph>=0.2.0"],
        "langchain": ["langchain>=0.1.0"],
        "full": ["langgraph>=0.2.0", "langchain>=0.1.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
)
