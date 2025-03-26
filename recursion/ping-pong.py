from Num_eights import num_eights
def pingpong(n):
    """Return the nth element of the ping-pong sequence.

    >>> pingpong(8)
    8
    >>> pingpong(10)
    6
    >>> pingpong(15)
    1
    >>> pingpong(21)
    -1
    >>> pingpong(22)
    -2
    >>> pingpong(30)
    -2
    >>> pingpong(68)
    0
    >>> pingpong(69)
    -1
    >>> pingpong(80)
    0
    >>> pingpong(81)
    1
    >>> pingpong(82)
    0
    >>> pingpong(100)
    -6
    >>> from construct_check import check
    >>> # ban assignment statements
    >>> check(HW_SOURCE_FILE, 'pingpong',
    ...       ['Assign', 'AnnAssign', 'AugAssign', 'NamedExpr'])
    True
    """
    "*** YOUR CODE HERE ***"
    def helper(index,value,direction):
        if index == n:
            return value
        elif index%8==0 or num_eights(str(index))>0:
            return helper(index+1,value-direction,-direction)
        else:
            return helper(index+1,value+direction,direction)
    return helper(1,1,1)
print(pingpong(int(input('input your number:'))))
# for i in range(1,20):
#     print(pingpong(i))