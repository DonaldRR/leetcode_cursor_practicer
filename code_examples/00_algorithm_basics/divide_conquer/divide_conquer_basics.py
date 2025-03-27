"""
分治法基础
========

分治法(Divide and Conquer)是一种算法设计策略，将问题划分为更小的子问题，
解决子问题后再将结果合并以解决原问题。

分治法通常包含三个步骤：
1. 分解（Divide）：将原问题分解为若干个规模较小的子问题
2. 解决（Conquer）：递归地解决各个子问题
3. 合并（Combine）：将子问题的解组合成原问题的解
"""

def explain_divide_conquer():
    """解释分治法的基本概念"""
    print("什么是分治法?")
    print("=" * 30)
    print("分治法(Divide and Conquer)是一种算法设计范式，包含三个步骤：")
    print("1. 分解：将原问题分解为若干个规模较小、相互独立的子问题")
    print("2. 解决：递归地解决各个子问题")
    print("3. 合并：将子问题的解组合成原问题的解")
    print()
    
    print("分治法的关键特点:")
    print("• 原问题可以分解为相似的子问题")
    print("• 子问题可以独立求解，子问题之间没有重叠")
    print("• 子问题的解可以合并为原问题的解")
    print("• 子问题足够小时，可以直接求解")
    print()
    
    print("分治法 vs 递归:")
    print("• 递归是一种编程技术，函数调用自身")
    print("• 分治法是一种算法设计策略，通常使用递归实现")
    print("• 分治法强调的是问题的分解和解的合并")
    print()

def divide_conquer_examples():
    """列举分治法的经典应用"""
    print("分治法的经典应用:")
    print("=" * 30)
    print("1. 归并排序")
    print("2. 快速排序")
    print("3. 二分搜索")
    print("4. 大整数乘法(Karatsuba算法)")
    print("5. 棋盘覆盖问题")
    print("6. 最近点对问题")
    print("7. 矩阵乘法(Strassen算法)")
    print()

def merge_sort(arr):
    """
    归并排序 - 分治法的经典应用
    
    时间复杂度: O(n log n)
    空间复杂度: O(n)
    """
    # 基本情况：数组长度 <= 1，已经排序
    if len(arr) <= 1:
        return arr
    
    # 分解步骤：将数组分成两半
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # 解决步骤：递归排序两半
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)
    
    # 合并步骤：合并两个已排序的子数组
    return merge(left_sorted, right_sorted)

def merge(left, right):
    """合并两个已排序的数组"""
    result = []
    i = j = 0
    
    # 比较两个数组的元素，按顺序合并
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 添加剩余元素
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def binary_search(arr, target, left=None, right=None):
    """
    二分搜索 - 分治法的应用
    
    时间复杂度: O(log n)
    空间复杂度: O(log n) - 递归版本
    """
    # 初始化左右边界
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
    
    # 基本情况：搜索区间为空
    if left > right:
        return -1
    
    # 分解步骤：找到中间元素
    mid = (left + right) // 2
    
    # 解决步骤：比较并决定搜索哪个子数组
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        # 搜索左半部分
        return binary_search(arr, target, left, mid - 1)
    else:
        # 搜索右半部分
        return binary_search(arr, target, mid + 1, right)

def max_subarray_sum(arr, left=None, right=None):
    """
    最大子数组和 - 分治解法 (LeetCode 53)
    
    时间复杂度: O(n log n)
    空间复杂度: O(log n) - 递归栈深度
    """
    # 初始化范围
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
    
    # 基本情况：单个元素
    if left == right:
        return arr[left]
    
    # 分解步骤：将数组分为两半
    mid = (left + right) // 2
    
    # 解决步骤：计算左右子数组的最大和
    left_max = max_subarray_sum(arr, left, mid)
    right_max = max_subarray_sum(arr, mid + 1, right)
    
    # 计算跨越中点的最大子数组和
    # 从中点向左扩展
    left_sum = 0
    left_border_max = float('-inf')
    for i in range(mid, left - 1, -1):
        left_sum += arr[i]
        left_border_max = max(left_border_max, left_sum)
    
    # 从中点向右扩展
    right_sum = 0
    right_border_max = float('-inf')
    for i in range(mid + 1, right + 1):
        right_sum += arr[i]
        right_border_max = max(right_border_max, right_sum)
    
    # 跨越中点的最大和
    cross_max = left_border_max + right_border_max
    
    # 合并步骤：返回三种情况中的最大值
    return max(left_max, right_max, cross_max)

def demo_merge_sort():
    """演示归并排序"""
    print("归并排序演示:")
    print("=" * 30)
    
    # 测试用例
    test_cases = [
        [5, 2, 4, 7, 1, 3, 2, 6],
        [38, 27, 43, 3, 9, 82, 10],
        [8, 7, 6, 5, 4, 3, 2, 1]
    ]
    
    for i, arr in enumerate(test_cases, 1):
        print(f"测试 {i}: {arr}")
        sorted_arr = merge_sort(arr)
        print(f"排序后: {sorted_arr}")
        print()
    
    # 性能测试
    import random
    import time
    
    sizes = [100, 1000, 10000]
    print("归并排序性能测试:")
    
    for size in sizes:
        # 生成随机数组
        random_array = [random.randint(1, 10000) for _ in range(size)]
        
        # 计时
        start_time = time.time()
        merge_sort(random_array)
        end_time = time.time()
        
        print(f"数组大小: {size}, 用时: {end_time - start_time:.6f}秒")
    
    print()

def demo_binary_search():
    """演示二分搜索"""
    print("二分搜索演示:")
    print("=" * 30)
    
    # 测试用例
    sorted_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    
    print(f"有序数组: {sorted_array}")
    test_targets = [7, 15, 20]
    
    for target in test_targets:
        index = binary_search(sorted_array, target)
        if index != -1:
            print(f"找到目标 {target} 在索引 {index} 处")
        else:
            print(f"未找到目标 {target}")
    
    print()

def demo_max_subarray():
    """演示最大子数组和"""
    print("最大子数组和演示 (LeetCode 53):")
    print("=" * 30)
    
    # 测试用例
    test_cases = [
        [-2, 1, -3, 4, -1, 2, 1, -5, 4],
        [5, 4, -1, 7, 8],
        [-1]
    ]
    
    for i, arr in enumerate(test_cases, 1):
        max_sum = max_subarray_sum(arr)
        print(f"测试 {i}: {arr}")
        print(f"最大子数组和: {max_sum}")
        print()
    
    # 比较分治法与动态规划
    print("分治法 vs 动态规划 (最大子数组和):")
    
    def max_subarray_dp(nums):
        """动态规划解法 O(n)"""
        if not nums:
            return 0
        
        current_sum = max_sum = nums[0]
        for num in nums[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    # 性能对比
    import time
    import random
    
    sizes = [100, 1000, 5000]
    
    for size in sizes:
        # 生成随机数组
        random_array = [random.randint(-100, 100) for _ in range(size)]
        
        # 分治法
        start_time = time.time()
        dc_result = max_subarray_sum(random_array)
        dc_time = time.time() - start_time
        
        # 动态规划
        start_time = time.time()
        dp_result = max_subarray_dp(random_array)
        dp_time = time.time() - start_time
        
        print(f"数组大小: {size}")
        print(f"分治法: {dc_result}, 用时: {dc_time:.6f}秒")
        print(f"动态规划: {dp_result}, 用时: {dp_time:.6f}秒")
        print(f"动态规划比分治法快 {dc_time/dp_time:.1f}倍")
        print()

def div_conq_analysis():
    """分析分治法的优缺点"""
    print("分治法的优缺点:")
    print("=" * 30)
    print("优点:")
    print("1. 适合并行计算")
    print("2. 对某些问题提供最优解")
    print("3. 直观易理解")
    print("4. 能有效解决复杂问题")
    print()
    
    print("缺点:")
    print("1. 递归调用有开销")
    print("2. 可能不如动态规划高效(针对有重叠子问题的情况)")
    print("3. 实现复杂度可能高")
    print("4. 合并步骤可能很复杂")
    print()
    
    print("分治法的适用场景:")
    print("1. 子问题相互独立(没有重叠)")
    print("2. 问题规模容易划分")
    print("3. 所有子问题具有相同的形式")
    print()

def leetcode_example():
    """LeetCode分治法示例: #215 数组中的第K个最大元素"""
    print("LeetCode示例: #215 数组中的第K个最大元素")
    print("=" * 30)
    print("问题: 在未排序的数组中找到第 k 个最大的元素。")
    print("请注意，它是排序后的第 k 个最大元素，而不是第 k 个不同的元素。")
    print()
    
    def find_kth_largest(nums, k):
        """
        使用分治法(快速选择)查找第K大元素
        时间复杂度: 平均O(n), 最坏O(n²)
        空间复杂度: O(log n)
        """
        # 将第K大转换为第(len(nums) - k)小
        return quick_select(nums, 0, len(nums) - 1, len(nums) - k)
    
    def quick_select(nums, left, right, k_smallest):
        """快速选择算法 - 分治法的应用"""
        # 基本情况
        if left == right:
            return nums[left]
        
        # 分解：选择一个pivot并分区
        pivot_index = partition(nums, left, right)
        
        # 根据pivot位置决定搜索哪一部分
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            # 在左侧继续搜索
            return quick_select(nums, left, pivot_index - 1, k_smallest)
        else:
            # 在右侧继续搜索
            return quick_select(nums, pivot_index + 1, right, k_smallest)
    
    def partition(nums, left, right):
        """将数组分区，返回pivot的索引"""
        # 选择最右边元素作为pivot
        pivot = nums[right]
        i = left
        
        # 小于pivot的放左边，大于pivot的放右边
        for j in range(left, right):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        
        # 将pivot放到正确位置
        nums[i], nums[right] = nums[right], nums[i]
        return i
    
    # 测试用例
    test_cases = [
        ([3, 2, 1, 5, 6, 4], 2),
        ([3, 2, 3, 1, 2, 4, 5, 5, 6], 4)
    ]
    
    for nums, k in test_cases:
        result = find_kth_largest(nums.copy(), k)
        print(f"数组: {nums}, k = {k}")
        print(f"第{k}大元素: {result}")
        print()
    
    # 比较分治法与排序方法
    import time
    import random
    print("分治法(快速选择) vs 排序方法:")
    
    def find_kth_largest_sorting(nums, k):
        """使用排序解决第K大问题"""
        return sorted(nums, reverse=True)[k-1]
    
    array_size = 10000
    random_array = [random.randint(1, 10000) for _ in range(array_size)]
    k = random.randint(1, array_size)
    
    # 分治法(快速选择)
    start_time = time.time()
    dc_result = find_kth_largest(random_array.copy(), k)
    dc_time = time.time() - start_time
    
    # 排序方法
    start_time = time.time()
    sort_result = find_kth_largest_sorting(random_array.copy(), k)
    sort_time = time.time() - start_time
    
    print(f"数组大小: {array_size}, k = {k}")
    print(f"分治法: {dc_result}, 用时: {dc_time:.6f}秒")
    print(f"排序法: {sort_result}, 用时: {sort_time:.6f}秒")
    print(f"结果一致: {dc_result == sort_result}")
    if dc_time < sort_time:
        print(f"分治法比排序快 {sort_time/dc_time:.1f}倍")
    else:
        print(f"排序比分治法快 {dc_time/sort_time:.1f}倍")

def main():
    """主函数，执行所有分治法示例和解释"""
    print("\n====== 分治法基础 ======\n")
    explain_divide_conquer()
    divide_conquer_examples()
    
    print("\n====== 分治法应用示例 ======\n")
    demo_merge_sort()
    demo_binary_search()
    demo_max_subarray()
    
    print("\n====== 分治法分析 ======\n")
    div_conq_analysis()
    
    print("\n====== LeetCode分治法示例 ======\n")
    leetcode_example()
    
    print("\n总结: 分治法是一种强大的算法设计策略，特别适合可以分解为独立子问题的场景。")
    print("理解并掌握分治法思想对于解决复杂问题非常重要。")

if __name__ == "__main__":
    main() 