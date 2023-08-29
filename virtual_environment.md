# Virtual environment

Python's official documentation says:

    "A virtual environment is a Python environment such that the Python interpreter, libraries and scripts installed into it are isolated from those installed in other virtual environments, and (by default) any libraries installed in a “system” Python, i.e., one which is installed as part of your operating system"

``` Your project becomes its own self contained application, independent of the system installed Python and its modules. ```

```
Creating a local development environment that is simple, repeatable, and powerful is an essential first step to developing a testable software project. This technique is also a skill that translates to any future software development project.
```
[Guide to python virtualenv ( `Hitchhiker's guide to Python`, chapter 1, O'Reilly)](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html)

Create a virtual environment with Makefile and a requirements.txt File.
## Installing a Virtual Environment

install virtualenv to your host Python : ``` pip install virtualenv ```

on mac OS X ```brew install autoenv```

create a new project folder, then run: ```python<version> -m venv <virtual-environment-name>```

for example :   ```python3.8 -m venv env```

place in project directory ```cd ft_linear_regression```

create a virtual environment ```virtualenv venv```

activate virtual environment
```source ./venv/bin/activate```

virtuel environment pip can install packages from a requirements.txt ```./venv/bin/pip install -r requirements.txt```

```
./venv/bin/pip install -U -r requirements.txt
./venv/bin/pip install -U seaborn
./venv/bin/python -m pip list
```

freeze packages current state ```pip freeze > requirements.txt```

deactivate virtual environment ```source ./venv/bin/deactivate```

remove ./venv ```rm -rf venv.```

## pip install options

```pip``` is the package installer for Python.

[pip install options](https://pip.pypa.io/en/stable/cli/pip_install/#options)

```pip install -r <file>```

-r, --requirement <file>

    Install from the given requirements file. This option can be used multiple times.

-U, --upgrade

    Upgrade all specified packages to the newest available version. 

-I, --ignore-installed

    Ignore the installed packages, overwriting them. This can break your system if the existing package is of a different version or was installed with a different package manager!

Run ```./venv/bin/python -m pip check``` to check for any broken requirement(s)

Run ```pip freeze > requirements.txt``` to update the Python requirements file.

[docs.python.org : venv](https://docs.python.org/fr/3/library/venv.html)

[virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)

[how to work in a virtual env](https://realpython.com/python-virtual-environments-a-primer/#how-can-you-work-with-a-python-virtual-environment)

## Requirements Files

![requirement files](https://pip.pypa.io/en/latest/user_guide/#requirements-files)

## Makefile

![Makefile tricks for Python projects](https://ricardoanderegg.com/posts/makefile-python-project-tricks/)