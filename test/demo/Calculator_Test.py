import unittest
from src.demo.Calculator import Calculator


class TestCalculation(unittest.TestCase):
    calculator = Calculator()

    def test_add(self):
        self.assertEqual(self.calculator.add(5, 7), 12)
        self.assertEqual(self.calculator.add(-3, 8), 5)
        self.assertEqual(self.calculator.add(-8, -15), -23)

    def test_substract(self):
        self.assertEqual(self.calculator.substract(5, 7), -2)
        self.assertEqual(self.calculator.substract(-3, 8), -11)
        self.assertEqual(self.calculator.substract(-8, -15), 7)

    def test_multiply(self):
        self.assertEqual(self.calculator.multiply(5, 7), 35)
        self.assertEqual(self.calculator.multiply(-3, 8), -24)
        self.assertEqual(self.calculator.multiply(-8, -15), 120)

    def test_divide(self):
        self.assertEqual(self.calculator.divide(4, 8), 0.5)
        self.assertEqual(self.calculator.divide(8, 2), 4.0)


if __name__ == '__main__':
    unittest.main()
