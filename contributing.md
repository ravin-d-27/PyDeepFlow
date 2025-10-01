# Contributing to PyDeepFlow

First off, thank you for considering contributing to PyDeepFlow! It's people like you that make PyDeepFlow such a great tool.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your_username/PyDeepFlow.git
    ```
3.  **Install the dependencies.** We recommend using a virtual environment:
    ```bash
    cd PyDeepFlow
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    pip install -e .[testing]
    ```

## Running Tests

To make sure everything is working correctly, please run the tests before submitting a pull request.

```bash
python -m unittest discover tests
```

## Submitting Changes

1.  Create a new branch for your changes:
    ```bash
    git checkout -b my-feature-branch
    ```
2.  Make your changes and commit them with a clear message.
3.  Push your branch to your fork:
    ```bash
    git push origin my-feature-branch
    ```
4.  Open a pull request on the [PyDeepFlow repository](https://github.com/ravin-d-27/PyDeepFlow/pulls).

## Code Style

Please try to follow the existing code style. We use `black` for code formatting.

## Reporting Bugs

If you find a bug, please open an issue on the [issue tracker](https://github.com/ravin-d-27/PyDeepFlow/issues). Please include as much information as possible, including:

*   A clear and descriptive title.
*   A detailed description of the bug.
*   Steps to reproduce the bug.
*   The expected behavior.
*   The actual behavior.
*   Your operating system and Python version.

## Suggesting Enhancements

If you have an idea for a new feature, please open an issue on the [issue tracker](https://github.com/ravin-d-27/PyDeepFlow/issues). Please include:

*   A clear and descriptive title.
*   A detailed description of the enhancement.
*   The motivation for the enhancement.

Thank you for your contribution!