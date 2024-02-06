import multiprocessing
import time


def read_matrix_from_file(filename):
    """Reads a matrix from a file."""
    with open(filename, 'r') as file:
        return [list(map(int, line.split())) for line in file]


def split_matrix(matrix, submatrix_size=500):
    """Splits a matrix into smaller submatrices."""
    submatrices = []
    for i in range(0, len(matrix), submatrix_size):
        for j in range(0, len(matrix[0]), submatrix_size):
            submatrix = [row[j:j + submatrix_size] for row in matrix[i:i + submatrix_size]]
            submatrices.append(submatrix)
    return submatrices


def display_submatrices(submatrices):
    """Displays the submatrices."""
    for idx, submatrix in enumerate(submatrices, start=1):
        print(f"Submatrix {idx}:")
        for row in submatrix:
            print(" ".join(map(str, row)))
        print("-" * 20)  # Separator for readability


def raise_elements_to_power(submatrix, power=20):
    """Raises each element in the submatrix to a specified power."""
    return [[element ** power for element in row] for row in submatrix]


def process_sequentially(submatrices, power=20):
    """Processes submatrices sequentially."""
    start_time = time.time()
    results = [raise_elements_to_power(submatrix, power) for submatrix in submatrices]
    return results, time.time() - start_time


def process_in_parallel(submatrices, power=20):
    """Processes submatrices in parallel using multiprocessing."""
    start_time = time.time()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.starmap(raise_elements_to_power, [(submatrix, power) for submatrix in submatrices])
    return results, time.time() - start_time


def main():
    matrix = read_matrix_from_file('input.txt')
    submatrices = split_matrix(matrix)

    _, time_sequential = process_sequentially(submatrices)
    print(f"Sequential processing time: {time_sequential:.4f} sec")

    _, time_parallel = process_in_parallel(submatrices)
    print(f"Parallel processing time: {time_parallel:.4f} sec")


if __name__ == '__main__':
    main()
