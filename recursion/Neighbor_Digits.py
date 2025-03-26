"""
Implement the function neighbor_digits. 
neighbor_digits takes in a positive integer num and an optional argument prev_digit. 
neighbor_digits outputs the number of digits in num that have the same digit to its right or left.
"""
def neighbor_digits(num, prev_digit=-1):
    """
    Returns the number of digits in num that have the same digit to its right
    or left.
    >>> neighbor_digits(111)
    3
    >>> neighbor_digits(123)
    0
    >>> neighbor_digits(112)
    2
    >>> neighbor_digits(1122)
    4
    """
    if len(num) == 0:
        return 0
    elif  num[0] == prev_digit or num[0] == num[1]:
        return 1 + neighbor_digits(num[1:], num[0])
    else:
        return neighbor_digits(num[1:], num[0])


print(neighbor_digits(input("Enter a number: ")))