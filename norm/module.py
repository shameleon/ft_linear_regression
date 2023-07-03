"""
this is module docstring
"""

def main(x):
    """
    this is docstring
    """
    return x * x

if __name__ == "__main__":
    main(5)
    print(main.__doc__)


"""
docstring

    A string literal which appears as the first expression in a class, function or module.
    While ignored when the suite is executed, it is recognized by the compiler 
    and put into the __doc__ attribute of the enclosing class, function or module. 
    Since it is available via introspection, it is the canonical place for documentation of the object.


    https://www.programiz.com/python-programming/docstrings
"""


# Correct:

# Aligned with opening delimiter.
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# Add 4 spaces (an extra level of indentation) to distinguish arguments from the rest.
def long_function_name(
        var_one, var_two, var_three,
        var_four):
    print(var_one)

# Hanging indents should add a level.
foo = long_function_name(
    var_one, var_two,
    var_three, var_four)

# Wrong:

# Arguments on first line forbidden when not using vertical alignment.
foo = long_function_name(var_one, var_two,
    var_three, var_four)

# Further indentation required as indentation is not distinguishable.
def long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)

