d = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,

}

def kwargstest(a, b, c,*args, **kwargs):
    print(locals())
    print(a, b, c)
    print(args)
    print(kwargs)

kwargstest(**d)