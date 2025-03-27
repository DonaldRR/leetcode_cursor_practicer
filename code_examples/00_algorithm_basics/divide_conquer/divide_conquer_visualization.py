"""
Divide and Conquer Approach: Merge Sort Example
==============================================

Divide and Conquer is an algorithm design paradigm with three main steps:
1. Divide: Break the original problem into smaller sub-problems
2. Conquer: Recursively solve these sub-problems
3. Combine: Merge the solutions of sub-problems to form the solution to the original problem

This file demonstrates the Divide and Conquer approach through a detailed implementation 
of Merge Sort, with visualization to help understanding the process.
"""

import time
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def print_title(title):
    """Print a styled title"""
    print("\n" + "=" * 60)
    print(f"{Fore.YELLOW}{title}")
    print("=" * 60)

def print_step(message, color=Fore.WHITE):
    """Print step information"""
    print(f"{color}{message}{Style.RESET_ALL}")

def print_array(arr, color=Fore.WHITE, highlight_indices=None, highlight_color=Fore.RED):
    """Print an array with optional highlighting for specific elements"""
    result = []
    
    for i, num in enumerate(arr):
        if highlight_indices and i in highlight_indices:
            result.append(f"{highlight_color}{num}{Style.RESET_ALL}")
        else:
            result.append(f"{color}{num}")
    
    print("[" + ", ".join(result) + f"{color}]")

def merge_sort_verbose(arr, depth=0, start_idx=0):
    """
    Merge Sort implementation with detailed step explanations
    """
    indent = "  " * depth
    n = len(arr)
    
    # Print the current array to be sorted
    print(f"{Fore.CYAN}{indent}Merge Sort: ", end="")
    print_array(arr)
    
    # Base case: array with length <= 1 is already sorted
    if n <= 1:
        print(f"{Fore.GREEN}{indent}Base case: Array length <= 1, already sorted")
        return arr
    
    # Divide step: Split the array into two halves
    mid = n // 2
    print(f"{Fore.MAGENTA}{indent}Divide: Split array into left part {arr[:mid]} and right part {arr[mid:]}")
    
    # Recursively sort the left half
    print(f"{Fore.BLUE}{indent}Recursively process left part:")
    left = merge_sort_verbose(arr[:mid], depth + 1, start_idx)
    
    # Recursively sort the right half
    print(f"{Fore.BLUE}{indent}Recursively process right part:")
    right = merge_sort_verbose(arr[mid:], depth + 1, start_idx + mid)
    
    # Combine step: Merge the sorted left and right halves
    print(f"{Fore.YELLOW}{indent}Merge: Combine sorted left part {left} and right part {right}")
    
    # Perform the merge operation
    result = []
    i = j = 0
    
    # Print the merging process
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            print(f"{Fore.WHITE}{indent}  Compare {left[i]} <= {right[j]}, choose left element {left[i]}")
            result.append(left[i])
            i += 1
        else:
            print(f"{Fore.WHITE}{indent}  Compare {left[i]} > {right[j]}, choose right element {right[j]}")
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    if i < len(left):
        print(f"{Fore.WHITE}{indent}  Add remaining left elements: {left[i:]}")
        result.extend(left[i:])
    if j < len(right):
        print(f"{Fore.WHITE}{indent}  Add remaining right elements: {right[j:]}")
        result.extend(right[j:])
    
    print(f"{Fore.GREEN}{indent}Merge result: ", end="")
    print_array(result)
    
    return result

def merge_sort(arr):
    """Standard merge sort implementation for actual sorting"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Visualize the merge sort process
def visualize_merge_sort():
    # Define data and global variables
    data = [38, 27, 43, 3, 9, 82, 10]
    states = []  # Store states during the sorting process
    
    def _merge_sort_viz(arr, start, end):
        """Merge sort implementation for visualization"""
        if end - start <= 1:
            return arr[start:end]
        
        mid = (start + end) // 2
        _merge_sort_viz(arr, start, mid)
        _merge_sort_viz(arr, mid, end)
        
        # Save state before merging, including start and end indices
        states.append((list(arr), start, mid, end))
        
        # Merge process
        merged = merge(arr[start:mid], arr[mid:end])
        arr[start:end] = merged
        
        # Save state after merging
        states.append((list(arr), start, mid, end))
        
        return merged
    
    # Execute merge sort and save states
    _merge_sort_viz(data.copy(), 0, len(data))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        arr, start, mid, end = states[frame]
        
        # Draw bar chart
        colors = ['#1f77b4'] * len(arr)
        
        # Highlight current operation range
        for i in range(start, end):
            if i < mid:
                colors[i] = '#ff7f0e'  # left part
            else:
                colors[i] = '#2ca02c'  # right part
        
        # Plot data
        bars = ax.bar(range(len(arr)), arr, color=colors)
        
        # Add labels
        ax.set_title(f'Merge Sort - Step {frame+1}: Merging [{start}:{mid}] and [{mid}:{end}]')
        ax.set_xlim(-1, len(arr))
        ax.set_ylim(0, max(arr) + 1)
        ax.set_xticks(range(len(arr)))
        
        return bars
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(states),
                                interval=1000, repeat=True)
    
    # Save animation as GIF
    ani.save('merge_sort_visualization.gif', writer='pillow', fps=1)
    
    print("Merge sort visualization saved as 'merge_sort_visualization.gif'")
    print("View with command: open merge_sort_visualization.gif")

# Another classic Divide and Conquer application: Quick Sort
def quick_sort_verbose(arr, depth=0, start=0):
    """Quick Sort implementation with detailed step explanations"""
    if len(arr) <= 1:
        if len(arr) == 1:
            print(f"{Fore.GREEN}{'  ' * depth}Base case: [{arr[0]}] already sorted")
        else:
            print(f"{Fore.GREEN}{'  ' * depth}Base case: [] already sorted")
        return arr
    
    print(f"{Fore.CYAN}{'  ' * depth}Quick Sort: {arr}")
    
    # Select pivot (here we simply choose the first element)
    pivot = arr[0]
    print(f"{Fore.MAGENTA}{'  ' * depth}Select pivot: {pivot}")
    
    # Partition process
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    
    print(f"{Fore.YELLOW}{'  ' * depth}Partition: ")
    print(f"{Fore.YELLOW}{'  ' * depth}Elements <= {pivot}: {less}")
    print(f"{Fore.YELLOW}{'  ' * depth}Elements > {pivot}: {greater}")
    
    # Recursively sort
    print(f"{Fore.BLUE}{'  ' * depth}Recursively process elements <= pivot:")
    sorted_less = quick_sort_verbose(less, depth + 1, start)
    
    print(f"{Fore.BLUE}{'  ' * depth}Recursively process elements > pivot:")
    sorted_greater = quick_sort_verbose(greater, depth + 1, start + len(less) + 1)
    
    # Combine results
    result = sorted_less + [pivot] + sorted_greater
    print(f"{Fore.GREEN}{'  ' * depth}Merge result: {result}")
    
    return result

# Another classic Divide and Conquer application: Binary Search
def binary_search_verbose(arr, target, left=0, right=None, depth=0):
    """Binary Search implementation with detailed step explanations"""
    if right is None:
        right = len(arr) - 1
    
    # Print current search range
    print(f"{Fore.CYAN}{'  ' * depth}Searching for {target} in index range [{left}:{right}]")
    print(f"{Fore.CYAN}{'  ' * depth}Current subarray: {arr[left:right+1]}")
    
    # Base case: empty search range
    if left > right:
        print(f"{Fore.RED}{'  ' * depth}Search range is empty, target not found")
        return -1
    
    # Find middle index
    mid = (left + right) // 2
    print(f"{Fore.MAGENTA}{'  ' * depth}Middle index: {mid}, middle value: {arr[mid]}")
    
    # Found the target
    if arr[mid] == target:
        print(f"{Fore.GREEN}{'  ' * depth}Found target {target} at index {mid}")
        return mid
    
    # Target is in the left half
    elif arr[mid] > target:
        print(f"{Fore.YELLOW}{'  ' * depth}Target {target} < middle value {arr[mid]}, search left half")
        return binary_search_verbose(arr, target, left, mid - 1, depth + 1)
    
    # Target is in the right half
    else:
        print(f"{Fore.YELLOW}{'  ' * depth}Target {target} > middle value {arr[mid]}, search right half")
        return binary_search_verbose(arr, target, mid + 1, right, depth + 1)

# Maximum Subarray Sum - Divide and Conquer Approach
def max_subarray_divide_conquer_verbose(arr, low=0, high=None, depth=0):
    """Maximum Subarray Sum using Divide and Conquer with detailed explanations"""
    indent = "  " * depth
    
    if high is None:
        high = len(arr) - 1
    
    print(f"{Fore.CYAN}{indent}Calculate maximum subarray sum for range [{low}:{high}]")
    print(f"{Fore.CYAN}{indent}Current subarray: {arr[low:high+1]}")
    
    # Base case: single element
    if low == high:
        print(f"{Fore.GREEN}{indent}Base case: single element {arr[low]}")
        return arr[low]
    
    # Find the middle point
    mid = (low + high) // 2
    print(f"{Fore.MAGENTA}{indent}Division point: index {mid}, value: {arr[mid]}")
    
    # Recursively find maximum subarray sum in left and right halves
    print(f"{Fore.BLUE}{indent}Recursively calculate left half [{low}:{mid}] maximum subarray sum:")
    left_max = max_subarray_divide_conquer_verbose(arr, low, mid, depth + 1)
    
    print(f"{Fore.BLUE}{indent}Recursively calculate right half [{mid+1}:{high}] maximum subarray sum:")
    right_max = max_subarray_divide_conquer_verbose(arr, mid + 1, high, depth + 1)
    
    # Find maximum subarray sum that crosses the middle
    print(f"{Fore.YELLOW}{indent}Calculate maximum subarray sum crossing the middle point:")
    
    # Find maximum sum in left half that extends to the middle
    left_sum = float('-inf')
    curr_sum = 0
    left_start_idx = mid
    
    for i in range(mid, low - 1, -1):
        curr_sum += arr[i]
        if curr_sum > left_sum:
            left_sum = curr_sum
            left_start_idx = i
    
    print(f"{Fore.WHITE}{indent}Maximum sum extending left from middle: {left_sum}, starting at index: {left_start_idx}")
    
    # Find maximum sum in right half that extends from the middle
    right_sum = float('-inf')
    curr_sum = 0
    right_end_idx = mid + 1
    
    for i in range(mid + 1, high + 1):
        curr_sum += arr[i]
        if curr_sum > right_sum:
            right_sum = curr_sum
            right_end_idx = i
    
    print(f"{Fore.WHITE}{indent}Maximum sum extending right from middle: {right_sum}, ending at index: {right_end_idx}")
    
    # Calculate cross sum
    cross_sum = left_sum + right_sum
    print(f"{Fore.WHITE}{indent}Maximum sum crossing middle: {cross_sum}")
    
    # Find maximum of the three sums
    print(f"{Fore.WHITE}{indent}Comparing three cases:")
    print(f"{Fore.WHITE}{indent}Left half maximum sum: {left_max}")
    print(f"{Fore.WHITE}{indent}Right half maximum sum: {right_max}")
    print(f"{Fore.WHITE}{indent}Cross middle maximum sum: {cross_sum}")
    
    max_sum = max(left_max, right_max, cross_sum)
    print(f"{Fore.GREEN}{indent}Return maximum value: {max_sum}")
    
    return max_sum

# Maximum Subarray Sum - Dynamic Programming Approach (for comparison)
def max_subarray_dp(arr):
    """Maximum Subarray Sum using Dynamic Programming"""
    print("\nSolving Maximum Subarray Sum problem using Dynamic Programming")
    print(f"Input array: {arr}")
    
    print("Dynamic Programming approach: For each position i, calculate the maximum subarray sum ending at i")
    
    curr_max = arr[0]
    global_max = arr[0]
    
    print(f"Initial state: dp[0] = {arr[0]} (single element case)")
    
    dp = [0] * len(arr)
    dp[0] = arr[0]
    
    for i in range(1, len(arr)):
        # DP transition: either start a new subarray or extend previous one
        dp[i] = max(arr[i], dp[i-1] + arr[i])
        print(f"dp[{i}] = max({arr[i]}, {dp[i-1]} + {arr[i]}) = {dp[i]}")
        
        global_max = max(global_max, dp[i])
    
    print(f"DP array: {dp}")
    print(f"Maximum subarray sum: {global_max}, ending at index {dp.index(global_max)}")
    
    return global_max

# Performance comparison of sorting algorithms
def compare_sorting_algorithms():
    """Compare the performance of Merge Sort with Python's built-in sort"""
    print_title("Sorting Algorithm Performance Comparison")
    
    # Test with different array sizes
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        # Generate random array
        arr = [random.randint(0, 1000) for _ in range(size)]
        
        # Test merge sort
        arr_copy = arr.copy()
        start_time = time.time()
        merge_sort(arr_copy)
        merge_sort_time = time.time() - start_time
        
        # Test Python's sort
        arr_copy = arr.copy()
        start_time = time.time()
        arr_copy.sort()
        python_sort_time = time.time() - start_time
        
        # Print results
        print(f"Array size {size}:")
        print(f"  Merge Sort: {merge_sort_time:.6f} seconds")
        print(f"  Python Sort: {python_sort_time:.6f} seconds")
        print(f"  Ratio: Merge Sort/Python Sort = {merge_sort_time/python_sort_time:.2f}")
        print()

# Explain Divide and Conquer thinking
def explore_divide_conquer_thinking():
    """Explore the Divide and Conquer algorithm design paradigm"""
    print_title("Divide and Conquer Thinking")
    
    print("What is Divide and Conquer?")
    print("Divide and Conquer is an algorithm design paradigm that breaks a problem into smaller subproblems,")
    print("solves them recursively, and then combines their solutions.")
    print()
    
    print("Core steps of Divide and Conquer:")
    print("1. Divide: Break the original problem into smaller subproblems")
    print("2. Conquer: Solve these subproblems recursively")
    print("3. Combine: Merge the solutions to form the answer to the original problem")
    print()
    
    print("When to use Divide and Conquer:")
    print("1. When the problem can be broken into similar, smaller subproblems")
    print("2. When subproblem solutions can be combined to solve the original problem")
    print("3. When subproblems are independent (no common subproblems, otherwise DP might be better)")
    print()
    
    print("Benefits of Divide and Conquer:")
    print("1. Solving complex problems by breaking them into simpler ones")
    print("2. Parallel computing: independent subproblems can be solved in parallel")
    print("3. Cache efficiency: some D&C algorithms make good use of memory caches")
    print()
    
    print("Classic Divide and Conquer algorithms:")
    print("1. Merge Sort: Divide array in half, sort each half, then merge")
    print("2. Quick Sort: Choose pivot, partition array, recursively sort partitions")
    print("3. Binary Search: Divide search space in half each time")
    print("4. Maximum Subarray Sum: Find max in left half, right half, and crossing the middle")
    print("5. Matrix Multiplication (Strassen's algorithm): Divide matrices into smaller ones")
    print("6. Closest Pair of Points: Find closest pair in a 2D plane")

# Explain Merge Sort complexity
def explain_merge_sort_complexity():
    """Explain the time and space complexity of Merge Sort"""
    print_title("Merge Sort Complexity Analysis")
    
    print("Time Complexity Analysis for Merge Sort:")
    print("Merge Sort's time complexity is O(n log n) because:")
    print("1. Division step: Each split reduces the problem size by half, resulting in log n levels")
    print("2. Merging step: Each level requires O(n) time for merging")
    print("Therefore total time complexity = levels × time per level = O(log n) × O(n) = O(n log n)")
    
    print("\nMerge Sort has a stable time complexity, always O(n log n) regardless of input,")
    print("which is an advantage over Quick Sort.")
    
    print("\nSpace Complexity Analysis for Merge Sort:")
    print("Merge Sort's space complexity is O(n) because:")
    print("1. It requires additional array space to store the merged results")
    print("2. Even though there are recursive calls, they all use the same auxiliary array space")
    
    print("\nThe disadvantage of Merge Sort is the extra space requirement,")
    print("compared to in-place sorting algorithms like Quick Sort.")
    print("However, Merge Sort is a stable sorting algorithm, meaning equal elements")
    print("maintain their relative order after sorting, which is an important advantage.")

def main():
    """Main function to demonstrate Divide and Conquer approach"""
    print_title("Divide and Conquer Approach - Detailed Explanation")
    
    # Explain Divide and Conquer thinking
    explore_divide_conquer_thinking()
    
    # Explain Merge Sort complexity
    explain_merge_sort_complexity()
    
    # Demonstrate Merge Sort
    print_title("Merge Sort Detailed Process Demonstration")
    arr = [38, 27, 43, 3, 9, 82, 10]
    print(f"Input array: {arr}")
    result = merge_sort_verbose(arr)
    print(f"Sorting result: {result}")
    
    # Visualize Merge Sort
    print_title("Merge Sort Visualization")
    visualize_merge_sort()
    
    # Demonstrate Quick Sort
    print_title("Quick Sort (another classic Divide and Conquer application)")
    arr = [38, 27, 43, 3, 9, 82, 10]
    print(f"Input array: {arr}")
    result = quick_sort_verbose(arr)
    print(f"Sorting result: {result}")
    
    # Demonstrate Binary Search
    print_title("Binary Search (another classic Divide and Conquer application)")
    arr = sorted([3, 9, 10, 27, 38, 43, 82])
    target = 27
    print(f"Searching for {target} in sorted array {arr}")
    result = binary_search_verbose(arr, target)
    print(f"Found target {target} at index {result}")
    
    # Demonstrate Maximum Subarray Sum
    print_title("Maximum Subarray Sum Problem (Divide and Conquer vs. Dynamic Programming)")
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"Input array: {arr}")
    
    print("\nUsing Divide and Conquer approach:")
    dc_result = max_subarray_divide_conquer_verbose(arr)
    
    print("\nUsing Dynamic Programming approach:")
    dp_result = max_subarray_dp(arr)
    
    print("\nComparing results:")
    print(f"Divide and Conquer result: {dc_result}")
    print(f"Dynamic Programming result: {dp_result}")
    print("Both methods yield the same result, but Dynamic Programming is typically more efficient")
    print("as it avoids repeated computations")
    
    # Compare sorting algorithm performance
    compare_sorting_algorithms()

if __name__ == "__main__":
    main() 