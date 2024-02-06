import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def read_matrix_from_file(filename):
    """Reads a matrix from a file."""
    with open(filename, 'r') as file:
        return [list(map(int, line.split())) for line in file]


def split_matrix_in_two(matrix):
    """Splits the matrix into two halves."""
    mid_index = len(matrix) // 2
    upper_half = matrix[:mid_index]
    lower_half = matrix[mid_index:]
    return upper_half, lower_half


def split_matrix(matrix, submatrix_size=750):
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
    """
    Processes submatrices in parallel using multiprocessing.

    Args:
    - submatrices: A list of submatrices (2D lists) to be processed.
    - power: The power to which each element in the submatrices will be raised. Defaults to 20.

    Returns:
    - A tuple containing two elements:
        1. The list of processed submatrices, where each element has been raised to the specified power.
        2. The total time taken to process all submatrices in parallel.
    """

    # Record the start time to calculate the total processing time later.
    start_time = time.time()

    # Create a pool of worker processes. The number of processes is set to the number of CPU cores available.
    # This helps in efficiently utilizing the available hardware resources.
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        # Use the pool's starmap method to apply the 'raise_elements_to_power' function to each submatrix in parallel.
        # 'starmap' is similar to the built-in map function but allows multiple arguments to be passed to the function being applied.
        # Here, each element in the iterable passed to starmap is a tuple containing a submatrix and the power.
        # The 'raise_elements_to_power' function is then called with these arguments for each submatrix.
        results = pool.starmap(raise_elements_to_power, [(submatrix, power) for submatrix in submatrices])

    # Calculate the total processing time by subtracting the start time from the current time.
    processing_time = time.time() - start_time

    # Return the processed submatrices and the total processing time.
    return results, processing_time


def process_in_threads(submatrices, power=20):
    """
    Processes submatrices in parallel using multithreading.

    Args:
    - submatrices: A list of submatrices (2D lists) to be processed.
    - power: The power to which each element in the submatrices will be raised. Defaults to 20.

    Returns:
    - A tuple containing two elements:
        1. The list of processed submatrices, where each element has been raised to the specified power.
        2. The total time taken to process all submatrices in parallel using threads.
    """

    # Record the start time to calculate the total processing time later.
    start_time = time.time()

    # Create a ThreadPoolExecutor as a context manager, specifying the max number of threads. 'max_workers' could be
    # set to the number of submatrices or another value based on your requirements and system capabilities.
    with ThreadPoolExecutor(max_workers=len(submatrices)) as executor:
        # Schedule the 'raise_elements_to_power' function to be executed on each submatrix in parallel.
        # 'submit' schedules the callable to be executed and returns a Future object.
        # A list comprehension is used to submit all tasks and collect the Future objects.
        futures = [executor.submit(raise_elements_to_power, submatrix, power) for submatrix in submatrices]

        # As the tasks complete, gather the results.
        # The 'result()' method on a Future object blocks until the callable completes, then returns the result.
        results = [future.result() for future in futures]

    # Calculate the total processing time.
    processing_time = time.time() - start_time

    # Return the processed submatrices and the total processing time.
    return results, processing_time


def process_in_two_threads(matrix, power=20):
    """Processes the two halves of the matrix in parallel using two threads."""
    start_time = time.time()

    upper_half, lower_half = split_matrix_in_two(matrix)

    # Function to process a matrix half
    def process_half(half):
        return [raise_elements_to_power(row, power) for row in half]

    # Create two threads for the two halves
    thread1 = threading.Thread(target=process_half, args=(upper_half,))
    thread2 = threading.Thread(target=process_half, args=(lower_half,))

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for both threads to complete
    thread1.join()
    thread2.join()

    end_time = time.time()
    print(f"Execution time with two threads: {end_time - start_time:.4f} seconds")
    

def process_in_two_processes(matrix, power=20):
    """Processes the two halves of the matrix in parallel using two processes."""
    start_time = time.time()

    upper_half, lower_half = split_matrix_in_two(matrix)

    # Function to process a matrix half in a separate process
    def process_half(half):
        return [raise_elements_to_power(row, power) for row in half]

    # Create a pool with two processes
    with multiprocessing.Pool(2) as pool:
        # Process each half in parallel
        pool.map(process_half, [upper_half, lower_half])

    end_time = time.time()
    print(f"Execution time with two processes: {end_time - start_time:.4f} seconds")



def main():
    matrix = read_matrix_from_file('input.txt')
    submatrices = split_matrix(matrix)

    # Single process
    single_process, time_sequential = process_sequentially(submatrices)
    print(f"Sequential processing time: {time_sequential:.4f} sec")

    # Max multiprocessing
    multi_process, time_parallel = process_in_parallel(submatrices)
    print(f"Parallel processing time: {time_parallel:.4f} sec")
    
    # Max multithreading
    multi_thread, time_parallel = process_in_threads(submatrices)
    print(f"Parallel threading time: {time_parallel:.4f} sec")

    # 2 Threads
    process_in_two_threads(matrix)

    # 2 Process
    process_in_two_processes(matrix)


if __name__ == '__main__':
    main()
