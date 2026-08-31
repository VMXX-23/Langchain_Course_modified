#MCP server implementation to add 2 nos
# math_server.py
from mcp.server.fastmcp import FastMCP
print("Starting math server...")
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of a number"""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@mcp.tool()
def power(a: int, b: int) -> float:
    """Raise a to the power of b"""
    return a ** b

@mcp.tool()
def root(a: int, b: int) -> float:
    """Calculate the b-th root of a
    For square root, b=2. For cube root / third root, b=3.
    """
    if a < 0 and b % 2 == 0:
        raise ValueError("Cannot take even root of a negative number.")
    return a ** (1 / b)

@mcp.tool()
def modulo(a: int, b: int) -> int:
    """Calculate a modulo b or Caculate the remainder of a divided by b"""
    if b == 0:
        raise ValueError("Cannot perform modulo by zero.")
    return a % b

@mcp.tool()
def gcd(a: int, b: int) -> int:
    """Calculate the greatest common divisor of a and b"""
    while b:
        a, b = b, a % b
    return abs(a)

@mcp.tool()
def lcm(a: int, b: int) -> int:
    """Calculate the least common multiple of a and b"""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)

@mcp.tool()
def is_prime(n: int) -> bool:
    """Check if a number is prime"""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

@mcp.tool()
def prime_factors(n: int) -> list:
    """Return the list of prime factors of a number"""
    factors = []
    # Check for number of 2s that divide n
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    # n must be odd at this point, so we can skip even numbers
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 2:
        factors.append(n)
    return factors

@mcp.tool()
def fibonacci(n: int) -> list:
    """Return the first n Fibonacci numbers"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        next_fib = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_fib)
    return fib_sequence

@mcp.tool()
def is_even(n: int) -> bool:
    """Check if a number is even"""
    return n % 2 == 0

@mcp.tool()
def is_odd(n: int) -> bool:
    """Check if a number is odd"""
    return n % 2 != 0

@mcp.tool()
def sum_of_digits(n: int) -> int:
    """Return the sum of the digits of a number"""
    return sum(int(digit) for digit in str(abs(n)))

@mcp.tool()
def reverse_number(n: int) -> int:
    """Return the reverse of a number"""
    sign = -1 if n < 0 else 1
    reversed_num = int(str(abs(n))[::-1])
    return sign * reversed_num

@mcp.tool()
def is_palindrome(n: int) -> bool:
    """Check if a number is a palindrome"""
    return str(n) == str(n)[::-1]

@mcp.tool()
def decimal_to_binary(n: int) -> str:
    """Convert a decimal number to binary"""
    if n < 0:
        raise ValueError("Cannot convert negative numbers to binary.")
    return bin(n)[2:]


if __name__ == "__main__":
    print("Server ready")
    mcp.run(transport="stdio")