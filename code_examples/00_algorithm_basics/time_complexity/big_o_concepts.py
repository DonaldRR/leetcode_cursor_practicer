"""
大O符号概念讲解
=============

本文件解释时间复杂度分析中的大O符号概念。
大O表示法描述了算法运行时间如何随输入规模变化而增长。
"""

# 常见的时间复杂度
TIME_COMPLEXITY_TYPES = [
    "O(1) - 常数时间复杂度",
    "O(log n) - 对数时间复杂度",
    "O(n) - 线性时间复杂度",
    "O(n log n) - 线性对数时间复杂度",
    "O(n²) - 平方时间复杂度",
    "O(n³) - 立方时间复杂度",
    "O(2^n) - 指数时间复杂度",
    "O(n!) - 阶乘时间复杂度"
]

def explain_big_o():
    """解释大O概念的基本含义"""
    print("大O表示法是什么?")
    print("=" * 30)
    print("大O表示法是用于描述算法时间复杂度的数学符号。")
    print("它表示算法运行时间增长的上限(最坏情况)。")
    print("大O忽略常数因子和低阶项，只关注增长最快的部分。")
    print()
    
    print("时间复杂度的排序(从最快到最慢):")
    for i, complexity in enumerate(TIME_COMPLEXITY_TYPES, 1):
        print(f"{i}. {complexity}")
    print()
    
    print("为什么大O表示法很重要?")
    print("=" * 30)
    print("1. 帮助我们理解算法效率")
    print("2. 预测算法在大规模输入下的性能")
    print("3. 比较不同算法的效率")
    print("4. 指导算法优化方向")
    print()

def big_o_rules():
    """大O表示法的基本规则"""
    print("大O表示法的基本规则:")
    print("=" * 30)
    print("1. 常数项可以忽略: O(2n) = O(n)")
    print("2. 低阶项可以忽略: O(n² + n) = O(n²)")
    print("3. 常数因子可以忽略: O(1/2 n²) = O(n²)")
    print("4. 算法复杂度通常由最深层嵌套循环决定")
    print()

def key_takeaways():
    """关于大O的关键知识点"""
    print("关键知识点:")
    print("=" * 30)
    print("• O(1): 无论输入大小，执行时间都一样")
    print("• O(log n): 每次迭代将问题规模缩小一定比例(如折半)")
    print("• O(n): 执行时间与输入大小成正比")
    print("• O(n log n): 许多高效排序算法的复杂度")
    print("• O(n²): 含有两层嵌套循环的算法")
    print("• O(2^n): 穷举所有可能组合")
    print("• 空间复杂度: 分析算法所需的额外空间")
    print()

def big_o_examples():
    """各种复杂度的简单例子"""
    print("复杂度示例代码:")
    print("=" * 30)
    print("O(1) - 常数时间:")
    print("   def get_first(arr):")
    print("       return arr[0]  # 无论数组多大，只需一步操作")
    print()
    
    print("O(log n) - 对数时间:")
    print("   def binary_search(arr, target):")
    print("       # 每次迭代将搜索范围减半")
    print("       left, right = 0, len(arr) - 1")
    print("       while left <= right:")
    print("           mid = (left + right) // 2")
    print("           if arr[mid] == target: return mid")
    print("           elif arr[mid] < target: left = mid + 1")
    print("           else: right = mid - 1")
    print("       return -1")
    print()
    
    print("O(n) - 线性时间:")
    print("   def linear_search(arr, target):")
    print("       # 最坏情况需要遍历整个数组")
    print("       for i, num in enumerate(arr):")
    print("           if num == target: return i")
    print("       return -1")
    print()
    
    print("O(n²) - 平方时间:")
    print("   def bubble_sort(arr):")
    print("       # 两层嵌套循环")
    print("       n = len(arr)")
    print("       for i in range(n):")
    print("           for j in range(0, n-i-1):")
    print("               if arr[j] > arr[j+1]:")
    print("                   arr[j], arr[j+1] = arr[j+1], arr[j]")
    print()
    
def main():
    """主函数，运行所有解释函数"""
    print("\n====== 大O表示法概念讲解 ======\n")
    explain_big_o()
    big_o_rules()
    key_takeaways()
    big_o_examples()
    print("\n总结: 理解时间复杂度是分析和优化算法的基础。")
    print("学会分析算法复杂度，可以帮助我们选择更高效的解决方案。")
    
if __name__ == "__main__":
    main() 