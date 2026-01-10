def count_common(list1, list2):
    count = 0
    for item in list1:
        if item in list2:
            count += 1
    return count


# Main program
list_a = [1, 2, 3, 4, 5]
list_b = [3, 4, 6, 7]

print("Number of common elements:", count_common(list_a, list_b))
