def has_subseq(n, seq):
    """
    Complete has_subseq, a function which takes in a number n and a "sequence"
    of digits seq and returns whether n contains seq as a subsequence, which
    does not have to be consecutive.

    >>> has_subseq(123, 12)
    True
    >>> has_subseq(141, 11)
    True
    >>> has_subseq(144, 12)
    False
    >>> has_subseq(144, 1441)
    False
    >>> has_subseq(1343412, 134)
    True
    """
    if len(seq) == 0:
        return True
    if len(n) == 0:
        return False
    
    return has_subseq(n[1:],seq[1:]) if n[0] == seq[0] else has_subseq(n[1:],seq)

print(has_subseq(input('1:'),input('2:')))