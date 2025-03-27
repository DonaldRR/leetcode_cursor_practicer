"""
分治法详解：以归并排序为例
=======================

分治法(Divide and Conquer)是一种解决复杂问题的算法思想，它的核心思路是：
1. 分解（Divide）：将原问题分解为若干个规模较小的子问题
2. 解决（Conquer）：递归地解决这些子问题
3. 合并（Combine）：将子问题的解合并构建原问题的解

本文件将通过归并排序详细展示分治法的思想和实现过程，并提供可视化帮助理解。
"""

import time
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from colorama import Fore, Style, init

# 初始化colorama
init(autoreset=True)

def print_title(title):
    """打印带有样式的标题"""
    print("\n" + "=" * 60)
    print(f"{Fore.YELLOW}{title}")
    print("=" * 60)

def print_step(message, color=Fore.WHITE):
    """打印步骤信息"""
    print(f"{color}{message}{Style.RESET_ALL}")

def print_array(arr, color=Fore.WHITE, highlight_indices=None, highlight_color=Fore.RED):
    """打印数组，可选择性高亮某些元素"""
    result = []
    
    for i, num in enumerate(arr):
        if highlight_indices and i in highlight_indices:
            result.append(f"{highlight_color}{num}{Style.RESET_ALL}")
        else:
            result.append(f"{color}{num}")
    
    print("[" + ", ".join(result) + f"{color}]")

def merge_sort_verbose(arr, depth=0, start_idx=0):
    """
    详细解释步骤的归并排序实现
    """
    indent = "  " * depth
    n = len(arr)
    
    # 打印当前层级的待排序数组
    print(f"{Fore.CYAN}{indent}归并排序: ", end="")
    print_array(arr)
    
    # 基本情况：数组长度 <= 1，已经排序
    if n <= 1:
        print(f"{Fore.GREEN}{indent}基本情况: 数组长度 <= 1，已排序")
        return arr
    
    # 分解步骤：将数组分成两半
    mid = n // 2
    print(f"{Fore.MAGENTA}{indent}分解: 将数组分为左半部分{arr[:mid]}和右半部分{arr[mid:]}")
    
    # 递归排序左半部分
    print(f"{Fore.BLUE}{indent}递归处理左半部分:")
    left = merge_sort_verbose(arr[:mid], depth + 1, start_idx)
    
    # 递归排序右半部分
    print(f"{Fore.BLUE}{indent}递归处理右半部分:")
    right = merge_sort_verbose(arr[mid:], depth + 1, start_idx + mid)
    
    # 合并步骤：将已排序的左右两部分合并
    print(f"{Fore.YELLOW}{indent}合并: 将已排序的左半部分{left}和右半部分{right}合并")
    
    # 执行合并
    result = []
    i = j = 0
    
    # 打印合并过程
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            print(f"{Fore.WHITE}{indent}  比较 {left[i]} <= {right[j]}, 选择左边元素 {left[i]}")
            result.append(left[i])
            i += 1
        else:
            print(f"{Fore.WHITE}{indent}  比较 {left[i]} > {right[j]}, 选择右边元素 {right[j]}")
            result.append(right[j])
            j += 1
    
    # 添加剩余元素
    if i < len(left):
        print(f"{Fore.WHITE}{indent}  添加左边剩余元素: {left[i:]}")
        result.extend(left[i:])
    if j < len(right):
        print(f"{Fore.WHITE}{indent}  添加右边剩余元素: {right[j:]}")
        result.extend(right[j:])
    
    print(f"{Fore.GREEN}{indent}合并结果: ", end="")
    print_array(result)
    
    return result

def merge_sort(arr):
    """标准归并排序实现，用于实际排序"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """合并两个已排序数组"""
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

# 可视化归并排序过程
def visualize_merge_sort():
    # 定义数据和全局变量
    data = [8, 3, 1, 5, 9, 2, 7, 4, 6]
    states = []  # 保存排序过程中的状态
    
    def _merge_sort_viz(arr, start, end):
        """用于生成可视化过程的归并排序"""
        if end - start <= 1:
            return arr[start:end]
        
        mid = (start + end) // 2
        _merge_sort_viz(arr, start, mid)
        _merge_sort_viz(arr, mid, end)
        
        # 保存开始合并前的状态，包括起始索引和结束索引
        states.append((list(arr), start, mid, end))
        
        # 合并过程
        merged = merge(arr[start:mid], arr[mid:end])
        arr[start:end] = merged
        
        # 保存合并后的状态
        states.append((list(arr), start, mid, end))
        
        return merged
    
    # 执行归并排序并保存状态
    _merge_sort_viz(data.copy(), 0, len(data))
    
    # 创建可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        arr, start, mid, end = states[frame]
        
        # 绘制条形图
        colors = ['#1f77b4'] * len(arr)
        
        # 高亮当前操作的范围
        for i in range(start, end):
            if i < mid:
                colors[i] = '#ff7f0e'  # 左半部分
            else:
                colors[i] = '#2ca02c'  # 右半部分
        
        # 绘制数据
        bars = ax.bar(range(len(arr)), arr, color=colors)
        
        # 添加标签
        ax.set_title(f'归并排序 - 第{frame+1}步: 合并 [{start}:{mid}] 和 [{mid}:{end}]')
        ax.set_xlim(-1, len(arr))
        ax.set_ylim(0, max(arr) + 1)
        ax.set_xticks(range(len(arr)))
        
        return bars
    
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(states),
                                interval=1000, repeat=True)
    
    # 保存动画为GIF
    ani.save('merge_sort_visualization.gif', writer='pillow', fps=1)
    
    print("归并排序可视化已保存为 'merge_sort_visualization.gif'")
    print("使用命令查看: open merge_sort_visualization.gif")

# 其他分治法经典应用：快速排序
def quick_sort_verbose(arr, depth=0, start=0):
    """详细解释步骤的快速排序实现"""
    if len(arr) <= 1:
        if len(arr) == 1:
            print(f"{Fore.GREEN}{'  ' * depth}基本情况: [{arr[0]}] 已排序")
        else:
            print(f"{Fore.GREEN}{'  ' * depth}基本情况: [] 已排序")
        return arr
    
    print(f"{Fore.CYAN}{'  ' * depth}快速排序: {arr}")
    
    # 选择基准元素（这里简单选择第一个元素）
    pivot = arr[0]
    print(f"{Fore.MAGENTA}{'  ' * depth}选择基准: {pivot}")
    
    # 分区过程
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    
    print(f"{Fore.YELLOW}{'  ' * depth}分区: ")
    print(f"{Fore.YELLOW}{'  ' * depth}小于等于 {pivot} 的元素: {less}")
    print(f"{Fore.YELLOW}{'  ' * depth}大于 {pivot} 的元素: {greater}")
    
    # 递归排序
    print(f"{Fore.BLUE}{'  ' * depth}递归处理小于等于基准的元素:")
    sorted_less = quick_sort_verbose(less, depth + 1, start)
    
    print(f"{Fore.BLUE}{'  ' * depth}递归处理大于基准的元素:")
    sorted_greater = quick_sort_verbose(greater, depth + 1, start + len(less) + 1)
    
    # 合并结果
    result = sorted_less + [pivot] + sorted_greater
    print(f"{Fore.GREEN}{'  ' * depth}合并结果: {result}")
    
    return result

# 其他分治法经典应用：二分查找
def binary_search_verbose(arr, target, left=0, right=None, depth=0):
    """详细解释步骤的二分查找实现"""
    if right is None:
        right = len(arr) - 1
    
    # 打印当前搜索范围
    print(f"{Fore.CYAN}{'  ' * depth}在索引范围 [{left}:{right}] 中查找 {target}")
    print(f"{Fore.CYAN}{'  ' * depth}当前子数组: {arr[left:right+1]}")
    
    # 基本情况：搜索范围为空
    if left > right:
        print(f"{Fore.RED}{'  ' * depth}搜索范围为空，目标不存在")
        return -1
    
    # 计算中间索引
    mid = (left + right) // 2
    print(f"{Fore.MAGENTA}{'  ' * depth}中间索引: {mid}, 中间值: {arr[mid]}")
    
    # 比较中间元素与目标
    if arr[mid] == target:
        print(f"{Fore.GREEN}{'  ' * depth}找到目标 {target} 在索引 {mid}")
        return mid
    elif arr[mid] > target:
        print(f"{Fore.YELLOW}{'  ' * depth}中间值 {arr[mid]} > 目标值 {target}，在左半部分查找")
        return binary_search_verbose(arr, target, left, mid - 1, depth + 1)
    else:
        print(f"{Fore.YELLOW}{'  ' * depth}中间值 {arr[mid]} < 目标值 {target}，在右半部分查找")
        return binary_search_verbose(arr, target, mid + 1, right, depth + 1)

# 分治法与动态规划对比：最大子数组和问题
def max_subarray_divide_conquer_verbose(arr, low=0, high=None, depth=0):
    """分治法解决最大子数组和问题"""
    if high is None:
        high = len(arr) - 1
    
    # 打印当前处理的数组
    print(f"{Fore.CYAN}{'  ' * depth}计算区间 [{low}:{high}] 的最大子数组和")
    print(f"{Fore.CYAN}{'  ' * depth}当前子数组: {arr[low:high+1]}")
    
    # 基本情况：只有一个元素
    if low == high:
        print(f"{Fore.GREEN}{'  ' * depth}基本情况: 单个元素 {arr[low]}")
        return arr[low]
    
    # 分解：计算中点
    mid = (low + high) // 2
    print(f"{Fore.MAGENTA}{'  ' * depth}分界点: 索引 {mid}, 值: {arr[mid]}")
    
    # 递归计算左半部分的最大子数组和
    print(f"{Fore.BLUE}{'  ' * depth}递归计算左半部分 [{low}:{mid}] 的最大子数组和:")
    left_max = max_subarray_divide_conquer_verbose(arr, low, mid, depth + 1)
    
    # 递归计算右半部分的最大子数组和
    print(f"{Fore.BLUE}{'  ' * depth}递归计算右半部分 [{mid+1}:{high}] 的最大子数组和:")
    right_max = max_subarray_divide_conquer_verbose(arr, mid + 1, high, depth + 1)
    
    # 计算跨越中点的最大子数组和
    print(f"{Fore.YELLOW}{'  ' * depth}计算跨越中点的最大子数组和:")
    
    # 计算包含中点和中点左边的最大和
    left_sum = float('-inf')
    curr_sum = 0
    for i in range(mid, low - 1, -1):
        curr_sum += arr[i]
        if curr_sum > left_sum:
            left_sum = curr_sum
            left_idx = i
    
    print(f"{Fore.YELLOW}{'  ' * depth}从中点向左的最大和: {left_sum}, 起始索引: {left_idx}")
    
    # 计算包含中点右边的最大和
    right_sum = float('-inf')
    curr_sum = 0
    for i in range(mid + 1, high + 1):
        curr_sum += arr[i]
        if curr_sum > right_sum:
            right_sum = curr_sum
            right_idx = i
    
    print(f"{Fore.YELLOW}{'  ' * depth}从中点向右的最大和: {right_sum}, 结束索引: {right_idx}")
    
    # 计算跨越中点的最大和
    cross_sum = left_sum + right_sum
    print(f"{Fore.YELLOW}{'  ' * depth}跨越中点的最大和: {cross_sum}")
    
    # 返回三种情况中的最大值
    print(f"{Fore.GREEN}{'  ' * depth}比较三种情况:")
    print(f"{Fore.GREEN}{'  ' * depth}左半部分最大和: {left_max}")
    print(f"{Fore.GREEN}{'  ' * depth}右半部分最大和: {right_max}")
    print(f"{Fore.GREEN}{'  ' * depth}跨越中点最大和: {cross_sum}")
    
    result = max(left_max, right_max, cross_sum)
    print(f"{Fore.GREEN}{'  ' * depth}返回最大值: {result}")
    
    return result

def max_subarray_dp(arr):
    """动态规划解决最大子数组和问题"""
    if not arr:
        return 0
    
    # 打印标题
    print(f"{Fore.CYAN}使用动态规划解决最大子数组和问题")
    print(f"{Fore.CYAN}输入数组: {arr}")
    print(f"{Fore.YELLOW}动态规划思路: 对于每个位置i，计算以i结尾的最大子数组和")
    
    # dp[i] 表示以索引i结尾的最大子数组和
    dp = [0] * len(arr)
    dp[0] = arr[0]
    
    print(f"{Fore.MAGENTA}初始状态: dp[0] = {dp[0]} (只有一个元素时，最大子数组和就是该元素)")
    
    # 填充dp数组
    for i in range(1, len(arr)):
        # 状态转移方程: dp[i] = max(arr[i], dp[i-1] + arr[i])
        # 要么只取当前元素，要么将当前元素加入前面的最大子数组
        dp[i] = max(arr[i], dp[i-1] + arr[i])
        print(f"{Fore.WHITE}dp[{i}] = max({arr[i]}, {dp[i-1]} + {arr[i]}) = {dp[i]}")
    
    # 找出dp数组中的最大值
    max_sum = max(dp)
    max_index = dp.index(max_sum)
    
    print(f"{Fore.GREEN}dp数组: {dp}")
    print(f"{Fore.GREEN}最大子数组和: {max_sum}，结束于索引 {max_index}")
    
    return max_sum

def compare_sorting_algorithms():
    """比较不同排序算法的性能"""
    print_title("排序算法性能比较")
    
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        # 生成随机数组
        arr = [random.randint(1, 1000) for _ in range(size)]
        
        # 测量归并排序性能
        start = time.time()
        merge_sort(arr.copy())
        merge_time = time.time() - start
        
        # 测量Python内置排序性能
        start = time.time()
        sorted(arr.copy())
        python_time = time.time() - start
        
        print(f"数组大小 {size}:")
        print(f"  归并排序: {merge_time:.6f} 秒")
        print(f"  Python排序: {python_time:.6f} 秒")
        print(f"  比例: 归并排序/Python排序 = {merge_time/python_time:.2f}")
        print()

def explore_divide_conquer_thinking():
    """探讨分治法思想"""
    print_title("分治法思维解析")
    
    print(f"{Fore.CYAN}什么是分治法？")
    print("分治法是一种算法设计范式，它通过递归地将问题分解为同类型的子问题，解决这些子问题，然后合并结果。")
    print("分治法的核心步骤:")
    print("1. 分解（Divide）: 将原问题分解为若干个规模较小的子问题")
    print("2. 解决（Conquer）: 递归地解决这些子问题")
    print("3. 合并（Combine）: 将子问题的解合并为原问题的解")
    print()
    
    print(f"{Fore.GREEN}分治法适用条件:")
    print("1. 问题可以分解为规模较小的相同问题")
    print("2. 子问题的解可以合并为原问题的解")
    print("3. 子问题相互独立，不包含公共子问题（如果有公共子问题，动态规划可能更合适）")
    print()
    
    print(f"{Fore.YELLOW}分治法的优势:")
    print("1. 解决复杂问题: 将复杂问题分解为简单问题，使问题更易于理解和解决")
    print("2. 并行计算: 子问题相互独立，可以并行处理，提高效率")
    print("3. 利用缓存: 对某些分治算法，可以缓存子问题结果以提高效率")
    print()
    
    print(f"{Fore.MAGENTA}分治法与递归的关系:")
    print("分治法依赖递归思想来解决问题，但不是所有递归都是分治法。")
    print("区别在于分治法特别强调将原问题分解为多个相同类型但规模较小的子问题，而一般递归可能只是简单地减小问题规模。")
    print()
    
    print(f"{Fore.CYAN}分治法经典应用:")
    print("1. 归并排序: 将数组分成两半，分别排序，然后合并")
    print("2. 快速排序: 选择基准，将数组分为小于和大于基准的两部分，递归排序")
    print("3. 二分查找: 将有序数组分成两半，根据目标与中间元素的大小关系决定在哪一半查找")
    print("4. 最大子数组和: 分别计算左半部分、右半部分和跨中点的最大子数组和")
    print("5. 矩阵乘法（Strassen算法）: 将矩阵分为更小的子矩阵进行乘法运算")
    print("6. 最近点对问题: 在二维平面上找到最近的两个点")

def explain_merge_sort_complexity():
    """解释归并排序的时间和空间复杂度"""
    print_title("归并排序复杂度分析")
    
    print(f"{Fore.CYAN}归并排序时间复杂度分析:")
    print("归并排序的时间复杂度是 O(n log n)，这是因为:")
    print("1. 分解步骤: 每次将问题规模减半，总共需要 log n 层")
    print("2. 合并步骤: 每层的合并操作需要 O(n) 时间")
    print("因此总时间复杂度 = 层数 × 每层时间 = O(log n) × O(n) = O(n log n)")
    print()
    
    print(f"{Fore.YELLOW}归并排序的时间复杂度是稳定的，无论输入如何，都是 O(n log n)，这一点优于快速排序。")
    print()
    
    print(f"{Fore.CYAN}归并排序空间复杂度分析:")
    print("归并排序的空间复杂度是 O(n)，这是因为:")
    print("1. 需要额外的数组空间来存储合并结果")
    print("2. 虽然有递归调用，但每层递归使用的是同一个辅助数组空间")
    print()
    
    print(f"{Fore.YELLOW}归并排序的缺点是需要额外的空间，这与原地排序算法如快速排序相比是一个劣势。")
    print("但归并排序是稳定的排序算法，即相等元素的相对顺序在排序后保持不变，这是它的一个重要优势。")

def main():
    """主函数"""
    print_title("分治法详解 - 以归并排序为例")
    
    # 探讨分治法思想
    explore_divide_conquer_thinking()
    
    # 归并排序复杂度分析
    explain_merge_sort_complexity()
    
    # 演示归并排序过程
    print_title("归并排序详细过程演示")
    arr = [38, 27, 43, 3, 9, 82, 10]
    print(f"{Fore.WHITE}输入数组: {arr}")
    sorted_arr = merge_sort_verbose(arr)
    print(f"{Fore.GREEN}排序结果: {sorted_arr}")
    
    # 可视化归并排序
    print_title("归并排序可视化")
    visualize_merge_sort()
    
    # 其他分治法经典应用：快速排序
    print_title("快速排序（分治法的另一个经典应用）")
    arr = [38, 27, 43, 3, 9, 82, 10]
    print(f"{Fore.WHITE}输入数组: {arr}")
    sorted_arr = quick_sort_verbose(arr)
    print(f"{Fore.GREEN}排序结果: {sorted_arr}")
    
    # 其他分治法经典应用：二分查找
    print_title("二分查找（分治法的另一个经典应用）")
    arr = [3, 9, 10, 27, 38, 43, 82]
    target = 27
    print(f"{Fore.WHITE}在有序数组 {arr} 中查找元素 {target}")
    result = binary_search_verbose(arr, target)
    if result != -1:
        print(f"{Fore.GREEN}找到目标 {target} 在索引 {result}")
    else:
        print(f"{Fore.RED}目标 {target} 不在数组中")
    
    # 分治法与动态规划对比：最大子数组和问题
    print_title("最大子数组和问题（分治法与动态规划对比）")
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"{Fore.WHITE}输入数组: {arr}")
    
    print("\n使用分治法:")
    max_sum_dc = max_subarray_divide_conquer_verbose(arr)
    
    print("\n使用动态规划:")
    max_sum_dp = max_subarray_dp(arr)
    
    print(f"\n{Fore.GREEN}对比结果:")
    print(f"{Fore.GREEN}分治法结果: {max_sum_dc}")
    print(f"{Fore.GREEN}动态规划结果: {max_sum_dp}")
    print(f"{Fore.YELLOW}两种方法得到相同的结果，但动态规划通常更高效，因为它避免了重复计算")
    
    # 比较排序算法性能
    compare_sorting_algorithms()

if __name__ == "__main__":
    main() 