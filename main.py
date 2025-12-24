def sum_numbers(*args):
    return sum(args)


def main():
    numbers = list(map(int, input("¬ведите числа через пробел: ").split()))
    print(sum_numbers(*numbers))


if __name__ == "__main__":
    main()
