from setuptools import setup, find_packages

setup(
    name="liars_dice_ai",
    version="0.1.0",
    description="A deep reinforcement learning implementation for the game Liar's Dice",
    author="Pratyush Singh",
    author_email="pratyush.kar.singh@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "tqdm>=4.62.0",
        "PyYAML>=5.4.1",
        "tensorboard>=2.7.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.9b0",
            "isort>=5.9.3",
            "flake8>=3.9.2",
            "mypy>=0.910"
        ]
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "train-liars-dice=src.scripts.train:main",
            "play-liars-dice=src.scripts.play:main"
        ]
    }
)