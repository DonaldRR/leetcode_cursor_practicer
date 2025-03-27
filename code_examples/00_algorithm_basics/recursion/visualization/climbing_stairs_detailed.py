"""
递归详解：以爬楼梯问题为例
=======================

LeetCode 70. 爬楼梯问题的详细递归解析。

问题描述：
假设你正在爬楼梯，需要n阶才能到达楼顶。
每次你可以爬1或2个台阶，问有多少种不同的方法可以爬到楼顶？

在这个文件中，我们将：
1. 详细分析递归解决该问题的思考过程
2. 展示递归调用树如何生成
3. 解释为什么简单递归会导致性能问题
4. 展示如何通过记忆化技术优化递归
5. 最后展示如何转换为迭代解法
"""

import time
import functools
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import networkx as nx

# 初始化colorama
init(autoreset=True)

def print_step(message, color=Fore.WHITE):
    """打印带有颜色的步骤信息"""
    print(f"{color}{message}{Style.RESET_ALL}")

def print_title(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"{Fore.YELLOW}{title}")
    print("=" * 60)

def print_function_call(function_name, args, depth=0):
    """打印函数调用信息"""
    indent = "  " * depth
    args_str = ", ".join(str(arg) for arg in args)
    print(f"{Fore.CYAN}{indent}调用: {function_name}({args_str})")

def print_return(value, depth=0):
    """打印函数返回值"""
    indent = "  " * depth
    print(f"{Fore.GREEN}{indent}返回: {value}")

def print_explanation(message, depth=0):
    """打印解释信息"""
    indent = "  " * depth
    print(f"{Fore.MAGENTA}{indent}解释: {message}")

def climb_stairs_verbose(n, depth=0):
    """
    爬楼梯问题的详细递归解法
    带有解释信息打印，帮助理解递归调用过程
    """
    print_function_call("climb_stairs_verbose", [n], depth)
    
    # 基本情况1: n = 1
    if n == 1:
        print_explanation("只有1个台阶，只能爬1步，所以只有1种方法", depth)
        print_return(1, depth)
        return 1
    
    # 基本情况2: n = 2
    if n == 2:
        print_explanation("有2个台阶，可以一次爬2步，或者分两次各爬1步，所以有2种方法", depth)
        print_return(2, depth)
        return 2
    
    # 递归情况
    print_explanation(f"对于{n}个台阶，我们考虑最后一步的走法:", depth)
    print_explanation(f"1. 如果最后一步走1个台阶，那么之前需要走完{n-1}个台阶", depth)
    print_explanation(f"2. 如果最后一步走2个台阶，那么之前需要走完{n-2}个台阶", depth)
    print_explanation(f"因此，总方法数 = 走完{n-1}个台阶的方法数 + 走完{n-2}个台阶的方法数", depth)
    
    # 第一个子问题：计算爬n-1个台阶的方法数
    print_explanation(f"计算爬{n-1}个台阶的方法数...", depth)
    ways_from_n_minus_1 = climb_stairs_verbose(n-1, depth+1)
    
    # 第二个子问题：计算爬n-2个台阶的方法数
    print_explanation(f"计算爬{n-2}个台阶的方法数...", depth)
    ways_from_n_minus_2 = climb_stairs_verbose(n-2, depth+1)
    
    # 合并子问题的结果
    total_ways = ways_from_n_minus_1 + ways_from_n_minus_2
    print_explanation(f"走完{n-1}个台阶的方法数({ways_from_n_minus_1}) + "
                     f"走完{n-2}个台阶的方法数({ways_from_n_minus_2}) = {total_ways}", depth)
    
    print_return(total_ways, depth)
    return total_ways

def climb_stairs_simple(n):
    """爬楼梯问题的简单递归解法"""
    if n <= 2:
        return n
    return climb_stairs_simple(n-1) + climb_stairs_simple(n-2)

# 使用functools.lru_cache来记忆化递归函数
@functools.lru_cache(maxsize=None)
def climb_stairs_memo(n):
    """爬楼梯问题的记忆化递归解法"""
    if n <= 2:
        return n
    return climb_stairs_memo(n-1) + climb_stairs_memo(n-2)

def climb_stairs_memo_manual(n, memo=None):
    """爬楼梯问题的手动实现记忆化递归解法"""
    if memo is None:
        memo = {}
    
    # 检查是否已经计算过
    if n in memo:
        return memo[n]
    
    # 基本情况
    if n <= 2:
        return n
    
    # 递归计算并存储结果
    memo[n] = climb_stairs_memo_manual(n-1, memo) + climb_stairs_memo_manual(n-2, memo)
    return memo[n]

def climb_stairs_iterative(n):
    """爬楼梯问题的迭代解法"""
    if n <= 2:
        return n
    
    # 初始化前两个状态
    a, b = 1, 2
    
    # 迭代计算后续状态
    for i in range(3, n+1):
        a, b = b, a + b
    
    return b

def validate_solutions(n):
    """验证不同解法的正确性"""
    print_title(f"验证n={n}的各种解法")
    
    # 计算标准答案（使用迭代方法作为基准）
    expected = climb_stairs_iterative(n)
    
    # 验证简单递归解法
    simple_result = climb_stairs_simple(n)
    print(f"简单递归: {simple_result}, {'正确' if simple_result == expected else '错误'}")
    
    # 验证记忆化递归解法
    memo_result = climb_stairs_memo(n)
    print(f"记忆化递归: {memo_result}, {'正确' if memo_result == expected else '错误'}")
    
    # 验证手动记忆化递归解法
    manual_memo_result = climb_stairs_memo_manual(n)
    print(f"手动记忆化递归: {manual_memo_result}, {'正确' if manual_memo_result == expected else '错误'}")
    
    # 验证迭代解法
    iterative_result = climb_stairs_iterative(n)
    print(f"迭代: {iterative_result}, {'正确' if iterative_result == expected else '错误'}")

def compare_performance():
    """比较不同解法的性能"""
    print_title("性能比较")
    
    test_cases = [10, 20, 30, 35]
    
    for n in test_cases:
        print(f"\n测试 n = {n}:")
        
        # 简单递归 - 只测试小数字，避免太慢
        if n <= 30:
            start_time = time.time()
            result = climb_stairs_simple(n)
            simple_time = time.time() - start_time
            print(f"简单递归: {result}, 用时: {simple_time:.6f}秒")
        else:
            print(f"简单递归: 跳过 (n太大)")
            simple_time = float('inf')
        
        # 记忆化递归
        start_time = time.time()
        result = climb_stairs_memo(n)
        memo_time = time.time() - start_time
        print(f"记忆化递归: {result}, 用时: {memo_time:.6f}秒")
        
        # 手动记忆化递归
        start_time = time.time()
        result = climb_stairs_memo_manual(n)
        manual_memo_time = time.time() - start_time
        print(f"手动记忆化递归: {result}, 用时: {manual_memo_time:.6f}秒")
        
        # 迭代
        start_time = time.time()
        result = climb_stairs_iterative(n)
        iterative_time = time.time() - start_time
        print(f"迭代: {result}, 用时: {iterative_time:.6f}秒")
        
        # 比较
        if n <= 30:
            print(f"记忆化递归比简单递归快 {simple_time/memo_time:.1f}倍")
        print(f"迭代比记忆化递归快 {memo_time/iterative_time:.1f}倍")

def visualize_recursion_tree(n=4):
    """可视化递归调用树"""
    print_title(f"递归调用树可视化 (n={n})")
    
    G = nx.DiGraph()
    calls = {}
    
    def build_tree(node, n):
        """构建递归调用树"""
        if n <= 2:
            return n
        
        left_child = f"F({n-1})"
        right_child = f"F({n-2})"
        
        # 添加边
        G.add_edge(node, left_child)
        G.add_edge(node, right_child)
        
        # 递归构建子树
        left_result = build_tree(left_child, n-1)
        right_result = build_tree(right_child, n-2)
        
        # 保存结果用于可视化
        calls[node] = left_result + right_result
        return calls[node]
    
    # 构建树
    root = f"F({n})"
    result = build_tree(root, n)
    calls[root] = result
    
    # 添加计算结果作为节点标签
    labels = {}
    for node in G.nodes():
        if node in calls:
            labels[node] = f"{node} = {calls[node]}"
        else:
            # 基本情况 (n=1 或 n=2)
            base_n = int(node.strip("F()"))
            labels[node] = f"{node} = {base_n}"
    
    # 布局
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    
    # 绘制图形
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color="lightblue", arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.title(f"递归调用树: 爬{n}阶楼梯")
    plt.axis("off")
    
    # 保存图像
    plt.savefig("climbing_stairs_recursion_tree.png")
    print(f"递归树图像已保存为 'climbing_stairs_recursion_tree.png'")
    print(f"如果要在终端中查看图像，可以使用: open climbing_stairs_recursion_tree.png")

def explore_recursive_thinking():
    """探讨递归思维方式"""
    print_title("递归思维解析")
    
    print(f"{Fore.CYAN}问题: 爬楼梯问题")
    print("有n阶楼梯，每次可以爬1或2阶，问有多少种方法可以爬到顶部?\n")
    
    print(f"{Fore.GREEN}1. 如何用递归思维分析这个问题?")
    print("递归思维的核心是将大问题分解为小问题，并找到基本情况。")
    print("对于爬楼梯问题，我们可以这样思考：")
    print("- 要爬到第n阶，最后一步可能是从第(n-1)阶爬1阶上来")
    print("- 或者从第(n-2)阶爬2阶上来")
    print("- 所以爬到第n阶的方法总数 = 爬到第(n-1)阶的方法数 + 爬到第(n-2)阶的方法数\n")
    
    print(f"{Fore.GREEN}2. 确定基本情况 (Base Cases)")
    print("递归必须有基本情况，否则会无限递归。对于这个问题：")
    print("- n=1: 只有1个台阶，只能爬1步，所以有1种方法")
    print("- n=2: 有2个台阶，可以一次爬2步，或者分两次各爬1步，所以有2种方法\n")
    
    print(f"{Fore.GREEN}3. 数学递推关系")
    print("从数学上，我们可以得到递推公式：")
    print("F(1) = 1")
    print("F(2) = 2")
    print("F(n) = F(n-1) + F(n-2) 当 n > 2")
    print("这实际上就是斐波那契数列！\n")
    
    print(f"{Fore.GREEN}4. 分析示例 n=4")
    print("让我们一步步分解 n=4 的情况:")
    print("F(4) = F(3) + F(2)")
    print("F(3) = F(2) + F(1)")
    print("F(2) = 2 (基本情况)")
    print("F(1) = 1 (基本情况)")
    print("回溯计算:")
    print("F(3) = F(2) + F(1) = 2 + 1 = 3")
    print("F(4) = F(3) + F(2) = 3 + 2 = 5")
    print("所以，爬4阶楼梯有5种不同的方法\n")
    
    print(f"{Fore.YELLOW}递归调用树的问题")
    print("简单递归会导致重复计算，例如计算F(5)时：")
    print("F(5) = F(4) + F(3)")
    print("F(4) = F(3) + F(2)")
    print("F(3) = F(2) + F(1)")
    print("注意F(3)在计算F(5)和F(4)时都被计算了一次，这是低效的\n")
    
    print(f"{Fore.GREEN}解决方案: 记忆化递归")
    print("使用哈希表存储已计算的结果，避免重复计算")
    print("当需要计算F(n)时，先检查是否已经计算过，如果是则直接返回结果\n")
    
    print(f"{Fore.GREEN}迭代解法")
    print("递归也可以转化为迭代，使用两个变量a和b分别代表F(n-2)和F(n-1):")
    print("从左到右计算，每次更新a和b，最终b就是答案")
    print("这种方法既高效又避免了递归调用栈的开销")

def step_by_step_example(n=3):
    """分步解析爬楼梯递归过程"""
    print_title(f"分步详解: n={n}的递归过程")
    
    print("下面，我们将详细跟踪整个递归过程中的每一步:")
    climb_stairs_verbose(n)

def explain_memoization():
    """详细解释记忆化技术"""
    print_title("记忆化递归详解")
    
    print("记忆化是一种优化递归的技术，通过存储已计算的结果避免重复计算。")
    
    print("\n手动实现记忆化的步骤:")
    print("1. 创建一个哈希表(字典)用于存储计算过的结果")
    print("2. 每次调用函数前，先检查结果是否已在哈希表中")
    print("3. 如果是，直接返回存储的结果")
    print("4. 如果不是，计算结果并存入哈希表，然后返回")
    
    print("\n示例代码:")
    print("""
def climb_stairs_memo_manual(n, memo=None):
    if memo is None:
        memo = {}
    
    # 检查是否已经计算过
    if n in memo:
        return memo[n]
    
    # 基本情况
    if n <= 2:
        return n
    
    # 递归计算并存储结果
    memo[n] = climb_stairs_memo_manual(n-1, memo) + climb_stairs_memo_manual(n-2, memo)
    return memo[n]
    """)
    
    print("\nPython中也可以使用内置的functools.lru_cache装饰器来自动实现记忆化:")
    print("""
@functools.lru_cache(maxsize=None)
def climb_stairs_memo(n):
    if n <= 2:
        return n
    return climb_stairs_memo(n-1) + climb_stairs_memo(n-2)
    """)
    
    n = 5
    print(f"\n以n={n}为例，我们来跟踪记忆化的过程:")
    memo = {}
    
    def track_memo(n, memo, depth=0):
        indent = "  " * depth
        print(f"{indent}计算F({n}), 当前memo: {memo}")
        
        if n in memo:
            print(f"{indent}F({n})在memo中找到: {memo[n]}")
            return memo[n]
        
        if n <= 2:
            memo[n] = n
            print(f"{indent}基本情况: F({n}) = {n}, 更新memo: {memo}")
            return memo[n]
        
        print(f"{indent}需要计算F({n-1})和F({n-2})")
        f_n_1 = track_memo(n-1, memo, depth+1)
        f_n_2 = track_memo(n-2, memo, depth+1)
        
        memo[n] = f_n_1 + f_n_2
        print(f"{indent}计算结果: F({n}) = F({n-1}) + F({n-2}) = {f_n_1} + {f_n_2} = {memo[n]}")
        print(f"{indent}更新memo: {memo}")
        return memo[n]
    
    result = track_memo(n, memo)
    print(f"\n最终结果: F({n}) = {result}")
    print(f"最终memo: {memo}")
    print("\n注意memo是如何避免重复计算的，每个F(i)只计算一次。")

def main():
    """主函数"""
    print_title("爬楼梯问题 - 递归详解")
    
    print("本教程将深入解析递归解决'爬楼梯'问题的思维过程")
    print("LeetCode 70: 假设你正在爬楼梯，需要n阶才能到达楼顶。")
    print("每次你可以爬1或2个台阶，问有多少种不同的方法可以爬到楼顶？")
    
    # 探讨递归思维
    explore_recursive_thinking()
    
    # 分步示例
    step_by_step_example(4)
    
    # 可视化递归树
    visualize_recursion_tree(5)
    
    # 解释记忆化
    explain_memoization()
    
    # 验证解法
    validate_solutions(6)
    
    # 性能比较
    compare_performance()

if __name__ == "__main__":
    main() 