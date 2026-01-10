def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        return None

    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            total = 0
            for k in range(len(B)):
                total += A[i][k] * B[k][j]
            row.append(total)
        result.append(row)

    return result


# Main program
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

output = matrix_multiply(A, B)
if output is None:
    print("Matrix multiplication not possible")
else:
    print("Product Matrix:", output)
