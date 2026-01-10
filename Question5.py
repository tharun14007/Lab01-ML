import random

def calculate_statistics():
    numbers = []
    for i in range(100):
        numbers.append(random.randint(100, 150))

    numbers.sort()

    mean = sum(numbers) / len(numbers)

    mid = len(numbers) // 2
    median = (numbers[mid] + numbers[mid - 1]) / 2

    freq = {}
    for num in numbers:
        if num in freq:
            freq[num] += 1
        else:
            freq[num] = 1

    mode = numbers[0]
    max_count = freq[mode]

    for key in freq:
        if freq[key] > max_count:
            max_count = freq[key]
            mode = key

    return mean, median, mode


# Main program
mean, median, mode = calculate_statistics()
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
