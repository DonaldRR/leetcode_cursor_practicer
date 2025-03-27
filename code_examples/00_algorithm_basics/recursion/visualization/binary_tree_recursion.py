"""
二叉树递归应用详解
=================

二叉树是最常见的树形数据结构，其特点是每个节点最多有两个子节点。
递归是处理二叉树问题的最自然和优雅的方法，本文件将详细介绍:

1. 二叉树的基本递归遍历（前序、中序、后序）
2. 递归解决二叉树的典型问题
3. 递归过程的可视化展示
4. 递归解与迭代解的对比
"""

import time
import random
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import networkx as nx

# 初始化colorama
init(autoreset=True)

class TreeNode:
    """二叉树节点定义"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        # 用于可视化的额外属性
        self.id = None  # 节点唯一标识
        self.depth = None  # 节点深度
        self.position = None  # 节点在图中的位置

def print_title(title):
    """打印带有样式的标题"""
    print("\n" + "=" * 60)
    print(f"{Fore.YELLOW}{title}")
    print("=" * 60)

def build_sample_tree():
    """构建示例二叉树
            1
           / \
          2   3
         / \   \
        4   5   6
           / \
          7   8
    """
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(6)
    root.left.right.left = TreeNode(7)
    root.left.right.right = TreeNode(8)
    return root

def build_tree_from_list(nums):
    """从列表构建二叉树（层序形式）
    例如：[1,2,3,4,5,null,6,null,null,7,8] 对应上面的示例树
    """
    if not nums or nums[0] is None:
        return None
    
    root = TreeNode(nums[0])
    queue = [root]
    i = 1
    
    while queue and i < len(nums):
        node = queue.pop(0)
        
        # 左子节点
        if i < len(nums) and nums[i] is not None:
            node.left = TreeNode(nums[i])
            queue.append(node.left)
        i += 1
        
        # 右子节点
        if i < len(nums) and nums[i] is not None:
            node.right = TreeNode(nums[i])
            queue.append(node.right)
        i += 1
    
    return root

def preorder_traversal_verbose(root, depth=0):
    """详细解释的前序遍历（根-左-右）"""
    indent = "  " * depth
    
    if not root:
        print(f"{Fore.CYAN}{indent}遇到空节点，返回空列表")
        return []
    
    print(f"{Fore.CYAN}{indent}前序遍历节点: {root.val}")
    print(f"{Fore.MAGENTA}{indent}当前节点值: {root.val}，加入结果列表")
    
    result = [root.val]
    
    print(f"{Fore.YELLOW}{indent}递归遍历左子树:")
    left_result = preorder_traversal_verbose(root.left, depth + 1)
    
    print(f"{Fore.GREEN}{indent}左子树遍历完成，结果: {left_result}")
    
    print(f"{Fore.YELLOW}{indent}递归遍历右子树:")
    right_result = preorder_traversal_verbose(root.right, depth + 1)
    
    print(f"{Fore.GREEN}{indent}右子树遍历完成，结果: {right_result}")
    
    print(f"{Fore.WHITE}{indent}合并结果: [当前节点值] + [左子树结果] + [右子树结果]")
    result = result + left_result + right_result
    
    print(f"{Fore.WHITE}{indent}节点 {root.val} 的遍历结果: {result}")
    return result

def inorder_traversal_verbose(root, depth=0):
    """详细解释的中序遍历（左-根-右）"""
    indent = "  " * depth
    
    if not root:
        print(f"{Fore.CYAN}{indent}遇到空节点，返回空列表")
        return []
    
    print(f"{Fore.CYAN}{indent}中序遍历节点: {root.val}")
    
    print(f"{Fore.YELLOW}{indent}递归遍历左子树:")
    left_result = inorder_traversal_verbose(root.left, depth + 1)
    
    print(f"{Fore.GREEN}{indent}左子树遍历完成，结果: {left_result}")
    print(f"{Fore.MAGENTA}{indent}当前节点值: {root.val}，加入结果列表")
    
    result = left_result + [root.val]
    
    print(f"{Fore.YELLOW}{indent}递归遍历右子树:")
    right_result = inorder_traversal_verbose(root.right, depth + 1)
    
    print(f"{Fore.GREEN}{indent}右子树遍历完成，结果: {right_result}")
    
    print(f"{Fore.WHITE}{indent}合并结果: [左子树结果] + [当前节点值] + [右子树结果]")
    result = result + right_result
    
    print(f"{Fore.WHITE}{indent}节点 {root.val} 的遍历结果: {result}")
    return result

def postorder_traversal_verbose(root, depth=0):
    """详细解释的后序遍历（左-右-根）"""
    indent = "  " * depth
    
    if not root:
        print(f"{Fore.CYAN}{indent}遇到空节点，返回空列表")
        return []
    
    print(f"{Fore.CYAN}{indent}后序遍历节点: {root.val}")
    
    print(f"{Fore.YELLOW}{indent}递归遍历左子树:")
    left_result = postorder_traversal_verbose(root.left, depth + 1)
    
    print(f"{Fore.GREEN}{indent}左子树遍历完成，结果: {left_result}")
    
    print(f"{Fore.YELLOW}{indent}递归遍历右子树:")
    right_result = postorder_traversal_verbose(root.right, depth + 1)
    
    print(f"{Fore.GREEN}{indent}右子树遍历完成，结果: {right_result}")
    
    print(f"{Fore.MAGENTA}{indent}当前节点值: {root.val}，加入结果列表")
    
    print(f"{Fore.WHITE}{indent}合并结果: [左子树结果] + [右子树结果] + [当前节点值]")
    result = left_result + right_result + [root.val]
    
    print(f"{Fore.WHITE}{indent}节点 {root.val} 的遍历结果: {result}")
    return result

def max_depth_recursive_verbose(root, depth=0, current_depth=0):
    """详细解释的计算二叉树最大深度的递归算法"""
    indent = "  " * depth
    
    if not root:
        print(f"{Fore.CYAN}{indent}遇到空节点，返回深度: {current_depth}")
        return current_depth
    
    # 当前节点的深度
    current_depth += 1
    print(f"{Fore.MAGENTA}{indent}访问节点: {root.val}, 当前深度: {current_depth}")
    
    # 递归计算左子树深度
    print(f"{Fore.YELLOW}{indent}递归计算左子树深度:")
    left_depth = max_depth_recursive_verbose(root.left, depth + 1, current_depth)
    print(f"{Fore.GREEN}{indent}左子树最大深度: {left_depth}")
    
    # 递归计算右子树深度
    print(f"{Fore.YELLOW}{indent}递归计算右子树深度:")
    right_depth = max_depth_recursive_verbose(root.right, depth + 1, current_depth)
    print(f"{Fore.GREEN}{indent}右子树最大深度: {right_depth}")
    
    # 取左右子树深度的最大值
    max_depth = max(left_depth, right_depth)
    print(f"{Fore.WHITE}{indent}节点 {root.val} 的最大深度: {max_depth}")
    
    return max_depth

def max_depth_iterative(root):
    """计算二叉树最大深度的迭代算法"""
    if not root:
        return 0
    
    queue = [(root, 1)]  # (节点, 深度)
    max_depth = 0
    
    while queue:
        node, depth = queue.pop(0)
        max_depth = max(max_depth, depth)
        
        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))
    
    return max_depth

def is_symmetric_recursive_verbose(root, depth=0):
    """详细解释的判断二叉树是否对称的递归算法"""
    indent = "  " * depth
    
    print(f"{Fore.CYAN}{indent}检查树是否对称")
    
    if not root:
        print(f"{Fore.GREEN}{indent}空树是对称的")
        return True
    
    print(f"{Fore.YELLOW}{indent}调用辅助函数检查左右子树是否镜像对称")
    
    def is_mirror(left, right, mirror_depth=0):
        mirror_indent = "  " * (depth + mirror_depth)
        
        # 两个都为空，对称
        if not left and not right:
            print(f"{Fore.GREEN}{mirror_indent}两个节点都为空，对称")
            return True
        
        # 一个为空一个不为空，不对称
        if not left or not right:
            print(f"{Fore.RED}{mirror_indent}一个节点为空一个不为空，不对称")
            return False
        
        print(f"{Fore.MAGENTA}{mirror_indent}比较节点: {left.val} 和 {right.val}")
        
        # 值不同，不对称
        if left.val != right.val:
            print(f"{Fore.RED}{mirror_indent}节点值不同: {left.val} != {right.val}，不对称")
            return False
        
        print(f"{Fore.WHITE}{mirror_indent}节点值相同: {left.val} == {right.val}")
        
        # 递归检查：左节点的左子树与右节点的右子树是否镜像对称
        print(f"{Fore.YELLOW}{mirror_indent}递归检查左节点的左子树与右节点的右子树是否镜像对称")
        outer_symmetric = is_mirror(left.left, right.right, mirror_depth + 1)
        
        if not outer_symmetric:
            print(f"{Fore.RED}{mirror_indent}外侧不对称")
            return False
        
        # 递归检查：左节点的右子树与右节点的左子树是否镜像对称
        print(f"{Fore.YELLOW}{mirror_indent}递归检查左节点的右子树与右节点的左子树是否镜像对称")
        inner_symmetric = is_mirror(left.right, right.left, mirror_depth + 1)
        
        if not inner_symmetric:
            print(f"{Fore.RED}{mirror_indent}内侧不对称")
            return False
        
        print(f"{Fore.GREEN}{mirror_indent}内外都对称")
        return True
    
    result = is_mirror(root.left, root.right)
    
    if result:
        print(f"{Fore.GREEN}{indent}树是对称的")
    else:
        print(f"{Fore.RED}{indent}树不是对称的")
    
    return result

def path_sum_recursive_verbose(root, target_sum, depth=0, current_path=None):
    """详细解释的路径总和递归算法
    判断是否存在根节点到叶子节点的路径，使得路径上所有节点值相加等于目标和
    """
    indent = "  " * depth
    
    if current_path is None:
        current_path = []
    
    if not root:
        print(f"{Fore.CYAN}{indent}遇到空节点，返回False")
        return False
    
    # 添加当前节点到路径
    current_path = current_path + [root.val]
    current_sum = sum(current_path)
    
    print(f"{Fore.MAGENTA}{indent}访问节点: {root.val}")
    print(f"{Fore.WHITE}{indent}当前路径: {current_path}, 当前和: {current_sum}, 目标和: {target_sum}")
    
    # 叶子节点，检查路径和是否等于目标和
    if not root.left and not root.right:
        result = current_sum == target_sum
        if result:
            print(f"{Fore.GREEN}{indent}找到目标路径: {current_path}, 和为: {current_sum}")
        else:
            print(f"{Fore.RED}{indent}路径和 {current_sum} != 目标和 {target_sum}")
        return result
    
    # 非叶子节点，继续搜索左右子树
    found = False
    
    if root.left:
        print(f"{Fore.YELLOW}{indent}搜索左子树:")
        left_result = path_sum_recursive_verbose(root.left, target_sum, depth + 1, current_path)
        if left_result:
            found = True
    
    # 如果左子树没找到，继续搜索右子树
    if not found and root.right:
        print(f"{Fore.YELLOW}{indent}搜索右子树:")
        right_result = path_sum_recursive_verbose(root.right, target_sum, depth + 1, current_path)
        if right_result:
            found = True
    
    if found:
        print(f"{Fore.GREEN}{indent}节点 {root.val} 子树中找到目标路径")
    else:
        print(f"{Fore.RED}{indent}节点 {root.val} 子树中没有找到目标路径")
    
    return found

def visualize_binary_tree(root, title="二叉树可视化"):
    """可视化二叉树结构"""
    if not root:
        return
    
    # 创建图并分配节点ID和深度
    G = nx.DiGraph()
    node_positions = {}
    node_labels = {}
    
    # 使用BFS为每个节点分配ID和计算位置
    queue = [(root, 0, 0, 0)]  # (node, id, depth, position)
    counter = 0
    
    while queue:
        node, node_id, depth, pos = queue.pop(0)
        if not node:
            continue
        
        # 保存节点属性
        node.id = node_id
        node.depth = depth
        node.position = pos
        
        # 为图添加节点
        G.add_node(node_id)
        node_positions[node_id] = (pos, -depth)  # x, y坐标
        node_labels[node_id] = str(node.val)
        
        # 处理左子节点
        if node.left:
            counter += 1
            left_id = counter
            left_pos = pos - 0.5 / (2 ** depth)
            G.add_edge(node_id, left_id)
            queue.append((node.left, left_id, depth + 1, left_pos))
        
        # 处理右子节点
        if node.right:
            counter += 1
            right_id = counter
            right_pos = pos + 0.5 / (2 ** depth)
            G.add_edge(node_id, right_id)
            queue.append((node.right, right_id, depth + 1, right_pos))
    
    # 绘制图形
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos=node_positions, labels=node_labels, with_labels=True,
            node_size=1500, node_color="lightblue", font_size=10, 
            font_weight="bold", arrows=False)
    
    plt.title(title)
    plt.axis("off")
    
    # 保存图像
    plt.savefig("binary_tree_visualization.png")
    print(f"二叉树图像已保存为 'binary_tree_visualization.png'")
    print(f"如果要在终端中查看图像，可以使用: open binary_tree_visualization.png")

def visualize_traversal(root, traversal_name, traversal_function):
    """可视化遍历过程"""
    if not root:
        return
    
    # 创建图并分配节点ID和深度
    G = nx.DiGraph()
    node_positions = {}
    node_labels = {}
    
    # 使用BFS为每个节点分配ID和计算位置
    queue = [(root, 0, 0, 0)]  # (node, id, depth, position)
    counter = 0
    
    while queue:
        node, node_id, depth, pos = queue.pop(0)
        if not node:
            continue
        
        # 保存节点属性
        node.id = node_id
        node.depth = depth
        node.position = pos
        
        # 为图添加节点
        G.add_node(node_id)
        node_positions[node_id] = (pos, -depth)  # x, y坐标
        node_labels[node_id] = str(node.val)
        
        # 处理左子节点
        if node.left:
            counter += 1
            left_id = counter
            left_pos = pos - 0.5 / (2 ** depth)
            G.add_edge(node_id, left_id)
            queue.append((node.left, left_id, depth + 1, left_pos))
        
        # 处理右子节点
        if node.right:
            counter += 1
            right_id = counter
            right_pos = pos + 0.5 / (2 ** depth)
            G.add_edge(node_id, right_id)
            queue.append((node.right, right_id, depth + 1, right_pos))
    
    # 获取遍历顺序
    traversal_result = traversal_function(root)
    
    # 创建遍历顺序映射到节点ID
    traversal_order = {}
    
    def fill_traversal_order(node, value_to_id):
        if not node:
            return
        value_to_id[node.val] = node.id
        fill_traversal_order(node.left, value_to_id)
        fill_traversal_order(node.right, value_to_id)
    
    value_to_id = {}
    fill_traversal_order(root, value_to_id)
    
    for i, val in enumerate(traversal_result):
        traversal_order[value_to_id[val]] = i + 1
    
    # 绘制图形
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos=node_positions, labels=node_labels, with_labels=True,
            node_size=1500, node_color="lightblue", font_size=10, 
            font_weight="bold", arrows=False)
    
    # 添加遍历顺序标签
    for node_id, order in traversal_order.items():
        x, y = node_positions[node_id]
        plt.text(x, y-0.1, f"({order})", fontsize=8, color="red",
                 ha="center", va="center")
    
    plt.title(f"{traversal_name}遍历顺序")
    plt.axis("off")
    
    # 保存图像
    filename = f"{traversal_name}_traversal.png"
    plt.savefig(filename)
    print(f"{traversal_name}遍历图像已保存为 '{filename}'")
    print(f"如果要在终端中查看图像，可以使用: open {filename}")

def explain_binary_tree_recursion():
    """解释二叉树递归的基本概念和应用"""
    print_title("二叉树递归基本概念")
    
    print(f"{Fore.CYAN}什么是二叉树递归?")
    print("二叉树递归是指利用递归方法处理二叉树问题，通常将大问题分解为针对左右子树的相同但规模更小的子问题。")
    print("二叉树天然具有递归结构：每个非叶子节点都可以看作是其子树的根节点。")
    print()
    
    print(f"{Fore.GREEN}二叉树递归的基本模式:")
    print("""
def solve_tree(root):
    # 基本情况（递归终止条件）
    if not root:
        return 某个基础值
        
    # 递归处理左子树
    left_result = solve_tree(root.left)
    
    # 递归处理右子树
    right_result = solve_tree(root.right)
    
    # 合并结果（根据不同问题有不同的合并方式）
    result = 某种方式合并(当前节点处理结果, left_result, right_result)
    
    return result
    """)
    print()
    
    print(f"{Fore.YELLOW}二叉树递归的优势:")
    print("1. 简洁性: 递归解法通常比迭代更简洁易懂")
    print("2. 自然性: 递归方式自然符合树的结构特性")
    print("3. 逻辑清晰: 问题分解为处理当前节点和递归处理子树")
    print()
    
    print(f"{Fore.RED}二叉树递归的局限性:")
    print("1. 调用栈开销: 对于非常深的树，可能导致栈溢出")
    print("2. 性能: 某些情况下（如遍历类问题），迭代解法可能更高效")
    print("3. 思维要求: 需要递归思维，可能不如迭代直观")
    print()
    
    print(f"{Fore.MAGENTA}常见的二叉树递归问题类型:")
    print("1. 遍历类: 前序、中序、后序、层序遍历")
    print("2. 属性计算类: 高度、深度、节点数、叶子数")
    print("3. 判定类: 是否平衡、是否对称、是否是BST")
    print("4. 路径问题: 路径和、最大路径、所有路径")
    print("5. 构造类: 从前序和中序构造树、从数组构造平衡树")

def compare_recursion_and_iteration():
    """比较递归和迭代解法的性能和特点"""
    print_title("递归与迭代解法对比")
    
    # 创建各种大小的二叉树进行测试
    trees = []
    for size in [10, 100, 1000, 10000]:
        values = list(range(1, size + 1))
        trees.append((size, build_tree_from_list(values)))
    
    print(f"{Fore.CYAN}二叉树最大深度计算 - 性能对比\n")
    print(f"{'树大小':>10} | {'递归时间 (秒)':>15} | {'迭代时间 (秒)':>15} | {'比例 (递归/迭代)':>20}")
    print("-" * 70)
    
    for size, tree in trees:
        # 测量递归方法时间
        start_time = time.time()
        recursive_depth = max_depth_recursive_verbose(tree, depth=0, current_depth=0)
        recursive_time = time.time() - start_time
        
        # 测量迭代方法时间
        start_time = time.time()
        iterative_depth = max_depth_iterative(tree)
        iterative_time = time.time() - start_time
        
        # 计算比例
        ratio = recursive_time / iterative_time if iterative_time > 0 else float('inf')
        
        print(f"{size:>10} | {recursive_time:>15.6f} | {iterative_time:>15.6f} | {ratio:>20.2f}")
    
    print("\n")
    print(f"{Fore.GREEN}递归解法优势:")
    print("1. 代码简洁度: 递归解法通常更简洁，易于理解")
    print("2. 问题分解: 自然符合'分治'思想，适合复杂树问题")
    print("3. 回溯能力: 适合需要回溯的问题（如路径查找）")
    print()
    
    print(f"{Fore.YELLOW}迭代解法优势:")
    print("1. 性能: 没有函数调用开销，通常更快")
    print("2. 内存使用: 避免调用栈开销，更适合处理大规模数据")
    print("3. 实时反馈: 可以随时中断或提供中间结果")
    print()
    
    print(f"{Fore.MAGENTA}选择建议:")
    print("1. 对时间和空间要求高的场景: 优先选择迭代")
    print("2. 代码可读性和维护性重要的场景: 优先选择递归")
    print("3. 树结构较复杂、问题难以直接拆解的场景: 优先选择递归")
    print("4. 树非常深的场景: 避免使用递归（防止栈溢出）")

def main():
    """主函数"""
    print_title("二叉树递归应用详解")
    
    # 解释二叉树递归概念
    explain_binary_tree_recursion()
    
    # 构建示例二叉树
    root = build_sample_tree()
    
    # 可视化二叉树
    visualize_binary_tree(root, "示例二叉树")
    
    # 演示前序遍历
    print_title("前序遍历详解（根-左-右）")
    result = preorder_traversal_verbose(root)
    print(f"\n前序遍历结果: {result}")
    
    # 可视化前序遍历
    visualize_traversal(root, "前序", lambda r: preorder_traversal_verbose(r))
    
    # 演示中序遍历
    print_title("中序遍历详解（左-根-右）")
    result = inorder_traversal_verbose(root)
    print(f"\n中序遍历结果: {result}")
    
    # 可视化中序遍历
    visualize_traversal(root, "中序", lambda r: inorder_traversal_verbose(r))
    
    # 演示后序遍历
    print_title("后序遍历详解（左-右-根）")
    result = postorder_traversal_verbose(root)
    print(f"\n后序遍历结果: {result}")
    
    # 可视化后序遍历
    visualize_traversal(root, "后序", lambda r: postorder_traversal_verbose(r))
    
    # 演示树的最大深度计算
    print_title("二叉树最大深度计算")
    depth = max_depth_recursive_verbose(root)
    print(f"\n树的最大深度: {depth}")
    
    # 演示对称树判断
    print_title("判断二叉树是否对称")
    
    # 创建一个对称的树
    symmetric_tree = TreeNode(1)
    symmetric_tree.left = TreeNode(2)
    symmetric_tree.right = TreeNode(2)
    symmetric_tree.left.left = TreeNode(3)
    symmetric_tree.left.right = TreeNode(4)
    symmetric_tree.right.left = TreeNode(4)
    symmetric_tree.right.right = TreeNode(3)
    
    visualize_binary_tree(symmetric_tree, "对称二叉树")
    is_symmetric = is_symmetric_recursive_verbose(symmetric_tree)
    print(f"\n对称树判断结果: {is_symmetric}")
    
    # 创建一个非对称的树
    non_symmetric_tree = TreeNode(1)
    non_symmetric_tree.left = TreeNode(2)
    non_symmetric_tree.right = TreeNode(2)
    non_symmetric_tree.left.right = TreeNode(3)
    non_symmetric_tree.right.right = TreeNode(3)
    
    visualize_binary_tree(non_symmetric_tree, "非对称二叉树")
    is_symmetric = is_symmetric_recursive_verbose(non_symmetric_tree)
    print(f"\n非对称树判断结果: {is_symmetric}")
    
    # 演示路径和问题
    print_title("路径和问题")
    
    # 创建一个用于路径和问题的树
    path_tree = TreeNode(5)
    path_tree.left = TreeNode(4)
    path_tree.right = TreeNode(8)
    path_tree.left.left = TreeNode(11)
    path_tree.left.left.left = TreeNode(7)
    path_tree.left.left.right = TreeNode(2)
    path_tree.right.left = TreeNode(13)
    path_tree.right.right = TreeNode(4)
    path_tree.right.right.right = TreeNode(1)
    
    visualize_binary_tree(path_tree, "路径和树")
    target = 22
    has_path = path_sum_recursive_verbose(path_tree, target)
    print(f"\n是否存在路径和为{target}的路径: {has_path}")
    
    # 递归与迭代对比
    compare_recursion_and_iteration()

if __name__ == "__main__":
    main() 