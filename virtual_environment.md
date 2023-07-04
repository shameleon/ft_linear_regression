# project setup

create a virtual environment with Makefile and a requirements.txt File.

## Virtual Environment

Python's official documentation says:

    "A virtual environment is a Python environment such that the Python interpreter, libraries and scripts installed into it are isolated from those installed in other virtual environments, and (by default) any libraries installed in a “system” Python, i.e., one which is installed as part of your operating system"

``` your project becomes its own self contained application, independent of the system installed Python and its modules. ```

 install venv to your host Python

```properties pip install virtualenv```

create a new project folder, then run:

```properties python<version> -m venv <virtual-environment-name>```
```properties python3.8 -m venv env```

```shell
cd ft_linear_regression
virtualenv env
source env/bin/activate
pip install pep8
```

![guide to python virtualenv](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html)
![docs.python.org : venv](https://docs.python.org/fr/3/library/venv.html)

![virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)

![how to work in a virtual env](https://realpython.com/python-virtual-environments-a-primer/#how-can-you-work-with-a-python-virtual-environment)

## Requirements Files

![requirement files](https://pip.pypa.io/en/latest/user_guide/#requirements-files)

## Makefile

![Makefile tricks for Python projects](https://ricardoanderegg.com/posts/makefile-python-project-tricks/)