import math

expression1 = math.log10(0.5 * (1 - 0.0909) / (0.0909 * (1 - 0.5)))
expression2 = 1 * 0 * math.log10(0.5)
print(f"1: {expression1}, 2: {expression2}")

result = expression1 + expression2

print(result)