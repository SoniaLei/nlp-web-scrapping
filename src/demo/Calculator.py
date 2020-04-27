class Calculator:

    def __init__(self):
        pass

    def add(self, x, y):
        return x + y

    def substract(self, x, y):
        return x - y

    def multiply(self, x, y):
        return x * y

    def divide(self, x, y):
        if y == 0:
            raise ValueError('Can not divide by zero')
        return x / y
