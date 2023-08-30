
From now on you must follow these
additional rules
• No code in the global scope. Use functions!
• Each program must have its main and not be a simple script:

```python
def main():
# your tests and your error handling
if __name__ == "__main__":
main()
```

• Any exception not caught will invalidate the exercices, even in the event of an error
that you were asked to test.
• All your functions must have a documentation (__doc__)
• Your code must be at the norm
◦ pip install flake8
◦ alias norminette=flake8

[PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)

[flake8](flake8)
[pycodestyle](https://pypi.org/project/pycodestyle/)

[coursera](https://www.coursera.org/learn/machine-learning)

## Make for Python project 
[makefile](https://earthly.dev/blog/python-makefile/)
Run ```python app.py```
install requirments ```pip install -r requirements.txt```
clean up .pyc files ```rm -rf __pycache__```
Creating a Virtual Environment ```python3 -m venv venv```

[makefile](https://medium.com/aigent/makefiles-for-python-and-beyond-5cf28349bf05)

## Python cheatsheet

[python cheatsheet](https://www.utc.fr/~jlaforet/Suppl/python-cheatsheets.pdf)

## Python Docstrings : The __doc__ attribute
Each Python object (functions, classes, variables,...) provides (if programmer has filled it) a short documentation which describes its features. You can access it with commands like print myobject.__doc__. You can provide a documentation for your own objects (functions for example) in the body of their definition as a string surrounded by three double-quotes:

```python
def myfunc():
    """'myfunc' documentation.
    
    Parameters : 
    None
    
    Return:
    None

    """

    pass

print(myfunc.__doc__)
```

## Annotations
[variable annotations](https://dzone.com/articles/python-3-variable-annotations)
```python
def add(a : int, b:int) -> int:
    return a + b

if __name__ == '__main__':
    add(2, 4)
```

[Type Annotations](https://runestone.academy/ns/books/published/fopp/Functions/TypeAnnotations.html)

```python
def count_words(text: str) -> dict[str, int]:
    words = text.split()
    d = {}
    for word in words:
        if word not in d:
            d[word] = 1
        else:
            d[word] += 1
    return d
```

## mutable objects

[passing mutable object](https://runestone.academy/ns/books/published/fopp/Functions/PassingMutableObjects.html)


## exceptions

```python
# program to print the reciprocal of even numbers

try:
    num = int(input("Enter a number: "))
    assert num % 2 == 0
except:
    print("Not an even number!")
else:
    reciprocal = 1/num
    print(reciprocal)
finally:
    print("This is finally block.")
```

## Bokeh

[Bokeh : la librairie Python de visualisation nouvelle génération](https://datascientest.com/bokeh-python-tout-savoir)

