# AutonoML

A framework for continuous automated machine learning.

## Installation

### Prerequisites

- **Anaconda**: If you don't have Anaconda installed, follow these instructions to [install Anaconda](https://www.anaconda.com/products/distribution).
- **Python**: Ensure that you have Python installed. You can check your Python version by running:
  ```bash
  python --version
  ```

### Procedure

1. Clone the repository.
  ```bash
  git clone https://github.com/UTS-CASLab/autonoml.git
  ```

2. Using Anaconda Prompt, navigate to the root directory of the cloned repository, i.e. where setup.py exists.

3. Create and activate a virtual environment. (Note: The current version of the code runs on Python 3.11.)
  ```bash
  conda create --name autonoml python=3.11
  
  conda activate autonoml
  ```

4. Install the AutonoML package. In the root folder (where setup.py is) run:
  ```bash
  pip install -e .
  ```
