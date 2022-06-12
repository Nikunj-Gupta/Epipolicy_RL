
class Expr:

    expr = []

    def __init__(self):
        pass

    def __getattr__(self, name):
        def _add(frm, to=None):
            self.expr.append(name + ":" + str(frm))
            if to is not None:
                self.expr.append(name + ":" + str(to))
            return self
        return _add

    def __str__(self):
        return ' '.join(self.expr)
