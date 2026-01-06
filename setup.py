from setuptools import setup, find_packages

setup(
    name="rh-agentic-operator",
    version="0.1.0",
    description="Kubernetes operator for managing BaseAgent custom resources",
    author="Red Hat",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "kopf>=1.37.0",
        "kubernetes>=28.1.0",
        "aiohttp>=3.9.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "rh-agentic-operator=rh_agentic_operator.operator:main",
        ],
    },
)



