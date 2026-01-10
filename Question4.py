def transpose(matrix):
    result = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        result.append(row)
    return result


# Main program
matrix = [[1, 2, 3],
          [4, 5, 6]]

print("Transpose of matrix:", transpose(matrix))
