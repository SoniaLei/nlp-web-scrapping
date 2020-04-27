def Add(a, b):
        print(a + b)
def AddList(a):
        s = 0
        for i in a:
                s = s + i
        print(s)
def Substract(a, b):
        print(a - b)
def Multiply(a, b):
        print(a * b)
def Divide(a, b):
        if b == 0:
                print("Error: Divide by zero")
        else:
                print(a / b)
def OddOrEven(a):
        if (a % 2) == 0:
                print("It is Even")
        else:
                print("It is Odd")
