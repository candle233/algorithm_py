def num_eights(pos):
    """Returns the number of times 8 appears as a digit of pos.

    >>> num_eights(3)
    0
    >>> num_eights(8)
    1
    >>> num_eights(88888888)
    8
    >>> num_eights(2638)
    1
    >>> num_eights(86380)
    2
    >>> num_eights(12345)
    0
    >>> from construct_check import check
    >>> # ban all assignment statements
    >>> check(HW_SOURCE_FILE, 'num_eights',
    ...       ['Assign', 'AnnAssign', 'AugAssign', 'NamedExpr'])
    True
    """
    # print('pos:',pos)
    if len(pos)==1:
        if pos =='8':
            return 1
        else:
            return 0
    else:
        return num_eights(pos[:(len(pos)//2)])+num_eights(pos[(len(pos)//2):])



# print(num_eights(input('input number:')))