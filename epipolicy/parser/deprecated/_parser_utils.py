PREFIX = 'x'

def isNumber(expr):
    return "numbers" in str(expr.func)

def isVariable(expr):
    return "Symbol" in str(expr.func)

def isAdd(expr):
    return "Add" in str(expr.func)

def isMul(expr):
    return "Mul" in str(expr.func)

def isPow(expr):
    return "Pow" in str(expr.func)
