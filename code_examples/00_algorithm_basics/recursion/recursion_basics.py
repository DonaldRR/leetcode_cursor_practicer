"""
递归基础概念
==========

递归是一种算法设计技巧，其中函数直接或间接地调用自身来解决问题。
递归通常用于解决可以分解为相似子问题的问题，是分治法的基础。

递归通常包含两个部分：
1. 基本情况（Base case）：不需要进一步递归调用就能直接解决的简单情况
2. 递归情况（Recursive case）：通过调用自身来解决的情况
"""

def explain_recursion():
    """解释递归的基本概念"""
    print("什么是递归?")
    print("=" * 30)
    print("递归是一种解决问题的方法，通过将问题分解为更小的同类子问题。")
    print("在编程中，递归函数是调用自身的函数。")
    print("递归解决方案包含两个关键部分：")
    print("1. 基本情况（终止条件）- 不再需要递归的简单情况")
    print("2. 递归情况 - 函数调用自身处理的情况")
    print()
    
    print("递归的思考方式:")
    print("=" * 30)
    print("设计递归算法需要'递归思维'，即考虑:")
    print("• 如何将原问题分解为更小的子问题")
    print("• 确定最简单的基本情况（什么时候停止递归）")
    print("• 如何合并子问题的解决方案")
    print()

def recursion_properties():
    """讨论递归的特性"""
    print("递归的关键特性:")
    print("=" * 30)
    print("1. 栈使用 - 每次递归调用都会在调用栈上创建新的栈帧")
    print("2. 内存开销 - 深度递归可能导致栈溢出")
    print("3. 可读性 - 通常使代码更简洁、易于理解")
    print("4. 性能 - 可能有额外开销，但某些问题的递归解决方案更自然")
    print()
    print("何时使用递归?")
    print("• 问题天然具有递归结构（如树的遍历）")
    print("• 问题可以分解为相似的子问题")
    print("• 代码简洁性比性能更重要的场景")
    print("• 回溯算法中")
    print()

def fibonacci_recursive(n):
    """
    使用递归计算斐波那契数列的第n项
    斐波那契数列: 0, 1, 1, 2, 3, 5, 8, 13, ...
    每个数是前两个数之和
    """
    # 基本情况
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # 递归情况
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_iterative(n):
    """使用迭代计算斐波那契数列的第n项（用于比较）"""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

def factorial_recursive(n):
    """
    递归计算阶乘: n! = n * (n-1) * ... * 2 * 1
    """
    # 基本情况
    if n == 0 or n == 1:
        return 1
    
    # 递归情况
    return n * factorial_recursive(n-1)

def sum_recursive(arr):
    """
    递归计算数组中所有元素的和
    """
    # 基本情况
    if len(arr) == 0:
        return 0
    
    # 递归情况
    return arr[0] + sum_recursive(arr[1:])

def compare_examples():
    """比较不同递归示例的运行情况"""
    print("递归示例:")
    print("=" * 30)
    
    # 斐波那契示例
    print("斐波那契数列示例:")
    for i in range(10):
        print(f"F({i}) = {fibonacci_recursive(i)}")
    
    # 阶乘示例
    print("\n阶乘示例:")
    for i in range(6):
        print(f"{i}! = {factorial_recursive(i)}")
    
    # 数组求和示例
    print("\n数组求和示例:")
    test_arrays = [
        [1, 2, 3, 4, 5],
        [10, 20, 30],
        [2, 4, 6, 8, 10]
    ]
    for arr in test_arrays:
        print(f"sum({arr}) = {sum_recursive(arr)}")
    
    # 性能比较
    import time
    print("\n递归与迭代性能比较 (斐波那契计算):")
    n = 30
    
    start = time.time()
    fib_rec = fibonacci_recursive(n)
    end = time.time()
    rec_time = end - start
    
    start = time.time()
    fib_iter = fibonacci_iterative(n)
    end = time.time()
    iter_time = end - start
    
    print(f"F({n}) = {fib_rec} (递归) - 用时: {rec_time:.6f}秒")
    print(f"F({n}) = {fib_iter} (迭代) - 用时: {iter_time:.6f}秒")
    print(f"迭代比递归快 {rec_time/iter_time:.1f}倍")

def recursion_pitfalls():
    """递归的常见陷阱和注意事项"""
    print("递归的陷阱与注意事项:")
    print("=" * 30)
    print("1. 栈溢出 - 递归太深会导致栈溢出错误")
    print("   解决方法: 确保递归有正确的基本情况，考虑尾递归优化")
    print()
    print("2. 重复计算 - 简单递归可能多次解决相同子问题")
    print("   解决方法: 使用记忆化（memoization）或动态规划")
    print()
    print("3. 性能问题 - 递归调用有额外开销")
    print("   解决方法: 某些情况下考虑迭代解法")
    print()
    print("4. 递归爆炸 - 如斐波那契递归产生指数级调用")
    print("   解决方法: 记忆化递归或自底向上的动态规划")

def leetcode_example():
    """LeetCode递归示例: #70 爬楼梯"""
    print("\nLeetCode递归示例: #70 爬楼梯")
    print("=" * 30)
    print("问题: 假设你正在爬楼梯，需要n步才能到达楼顶。")
    print("每次你可以爬1或2个台阶，问有多少种不同的方法可以爬到楼顶?")
    print()
    
    def climb_stairs_recursive(n):
        """简单递归解法 (会超时)"""
        if n <= 2:
            return n
        return climb_stairs_recursive(n-1) + climb_stairs_recursive(n-2)
    
    def climb_stairs_memoization(n):
        """带记忆化的递归解法"""
        memo = {1: 1, 2: 2}
        
        def dp(i):
            if i in memo:
                return memo[i]
            memo[i] = dp(i-1) + dp(i-2)
            return memo[i]
        
        return dp(n)
    
    def climb_stairs_iterative(n):
        """迭代解法"""
        if n <= 2:
            return n
        
        a, b = 1, 2
        for _ in range(3, n+1):
            a, b = b, a + b
        return b
    
    # 测试爬楼梯示例
    test_cases = [2, 3, 4, 5, 10]
    
    print("使用不同方法计算爬楼梯问题:")
    for n in test_cases:
        # 使用记忆化而不是简单递归，避免大n值时计算过慢
        memo_result = climb_stairs_memoization(n)
        iter_result = climb_stairs_iterative(n)
        print(f"n = {n}: {memo_result} 种方法 (记忆化递归 = 迭代: {memo_result == iter_result})")
    
    print("\n大数据对比 (n = 30):")
    import time
    
    # 记忆化方法
    start = time.time()
    memo_result = climb_stairs_memoization(30)
    memo_time = time.time() - start
    
    # 迭代方法
    start = time.time()
    iter_result = climb_stairs_iterative(30)
    iter_time = time.time() - start
    
    print(f"记忆化递归: {memo_result} (用时: {memo_time:.6f}秒)")
    print(f"迭代方法: {iter_result} (用时: {iter_time:.6f}秒)")
    
    print("\n简单递归与记忆化递归比较:")
    # 只测试小数据，避免简单递归超时
    n = 10
    
    start = time.time()
    rec_result = climb_stairs_recursive(n)
    rec_time = time.time() - start
    
    start = time.time()
    memo_result = climb_stairs_memoization(n)
    memo_time = time.time() - start
    
    print(f"n = {n}")
    print(f"简单递归: {rec_result} (用时: {rec_time:.6f}秒)")
    print(f"记忆化递归: {memo_result} (用时: {memo_time:.6f}秒)")
    print(f"记忆化比简单递归快 {rec_time/memo_time:.1f}倍")

def main():
    """主函数，执行所有递归示例和解释"""
    print("\n====== 递归基础概念 ======\n")
    explain_recursion()
    recursion_properties()
    print("\n====== 递归示例 ======\n")
    compare_examples()
    print("\n====== 递归注意事项 ======\n")
    recursion_pitfalls()
    leetcode_example()
    
    print("\n总结: 递归是一种强大的问题解决工具，但需要谨慎使用。")
    print("理解递归思想对于许多高级算法（如分治法、回溯等）是必不可少的基础。")

if __name__ == "__main__":
    main() 