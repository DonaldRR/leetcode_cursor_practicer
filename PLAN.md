# LeetCode 学习计划

## 编程语言与教学方法

- **教学语言**：Python（初学者友好、语法简洁、丰富的标准库）
- **教学方式**：知识点与代码结合教学，采用"理论+实践"并重的方式
- **代码实现**：每个知识点创建单独的.py文件，文件内包含：
  1. 知识点理论介绍与说明
  2. 相关题目设计与实现（最多2个LeetCode题目）
  3. 测试用例与结果展示
  4. 扩展练习与挑战
- **学习流程**：知识点讲解 → 设置题目 → 代码实现 → 测试结果 → 下一知识点
- **文件组织**：使用结构化目录，将基础知识与LeetCode题目分开存放

## 文件目录结构

```
.
├── PLAN.md                     # 学习计划文件
├── PRACTICE_LOG.md             # 练习记录文件
└── code_examples/              # 代码示例根目录
    ├── 00_algorithm_basics/    # 算法基础
    │   ├── time_complexity/    # 时间复杂度分析
    │   │   ├── big_o_concepts.py        # 大O概念讲解
    │   │   └── complexity_examples.py    # 复杂度示例分析
    │   ├── recursion/         # 递归
    │   │   ├── recursion_basics.py      # 递归基础概念
    │   │   └── recursion_applications.py # 递归应用
    │   ├── iteration/         # 迭代
    │   │   └── iteration_basics.py      # 迭代基础与应用
    │   └── divide_conquer/    # 分治法
    │       └── divide_conquer_basics.py # 分治法基础
    ├── 01_data_structures/     # 数据结构基础
    │   ├── arrays/            # 数组相关
    │   │   ├── array_basics.py          # 数组基础
    │   │   └── array_operations.py      # 数组操作
    │   ├── linked_lists/      # 链表相关
    │   │   ├── singly_linked_list.py    # 单链表
    │   │   └── doubly_linked_list.py    # 双链表
    │   ├── stacks/            # 栈相关
    │   │   └── stack_implementation.py  # 栈实现
    │   ├── queues/            # 队列相关
    │   │   └── queue_implementation.py  # 队列实现
    │   ├── hash_tables/       # 哈希表相关
    │   │   └── hash_table_implementation.py # 哈希表实现
    │   ├── trees/             # 树相关
    │   │   ├── binary_tree_basics.py    # 二叉树基础
    │   │   ├── binary_search_tree.py    # 二叉搜索树
    │   │   └── tree_traversal.py        # 树的遍历
    │   └── graphs/            # 图相关
    │       ├── graph_representation.py  # 图的表示
    │       └── graph_traversal.py       # 图的遍历
    ├── 02_array_string_problems/ # 数组与字符串问题
    │   ├── two_sum.py         # LeetCode 1: 两数之和
    │   ├── remove_duplicates.py # LeetCode 26: 删除排序数组中的重复项
    │   ├── longest_common_prefix.py # LeetCode 14: 最长公共前缀
    │   └── find_index.py      # LeetCode 28: 找出字符串中第一个匹配项的下标
    ├── 03_linked_list_problems/ # 链表问题
    │   ├── merge_sorted_lists.py # LeetCode 21: 合并两个有序链表
    │   ├── reverse_linked_list.py # LeetCode 206: 反转链表
    │   └── linked_list_cycle.py # LeetCode 141: 环形链表
    ├── 04_stack_queue_problems/ # 栈和队列问题
    │   ├── valid_parentheses.py # LeetCode 20: 有效的括号
    │   ├── implement_stack_using_queue.py # LeetCode 225: 用队列实现栈
    │   └── min_stack.py       # LeetCode 155: 最小栈
    ├── 05_hash_table_problems/ # 哈希表问题
    │   ├── group_anagrams.py   # LeetCode 49: 字母异位词分组
    │   ├── longest_consecutive.py # LeetCode 128: 最长连续序列
    │   └── two_sum.py          # LeetCode 1: 两数之和(哈希表解法)
    ├── 06_tree_problems/      # 树问题
    │   ├── inorder_traversal.py # LeetCode 94: 二叉树的中序遍历
    │   ├── max_depth.py        # LeetCode 104: 二叉树的最大深度
    │   ├── symmetric_tree.py   # LeetCode 101: 对称二叉树
    │   └── path_sum.py         # LeetCode 112: 路径总和
    ├── 07_graph_problems/     # 图问题
    │   ├── number_of_islands.py # LeetCode 200: 岛屿数量
    │   ├── course_schedule.py  # LeetCode 207: 课程表
    │   └── clone_graph.py      # LeetCode 133: 克隆图
    ├── 08_search_algorithms/  # 搜索算法
    │   ├── dfs/               # 深度优先搜索
    │   │   └── dfs_basics.py  # DFS基础与应用
    │   ├── bfs/               # 广度优先搜索
    │   │   └── bfs_basics.py  # BFS基础与应用
    │   └── binary_search/     # 二分查找
    │       ├── binary_search_basics.py  # 二分查找基础
    │       └── rotated_array_search.py  # LeetCode 33: 搜索旋转排序数组
    ├── 09_dynamic_programming/ # 动态规划
    │   ├── dp_introduction/    # 动态规划入门
    │   │   ├── dp_concepts.py  # 动态规划基本概念
    │   │   ├── climbing_stairs.py # LeetCode 70: 爬楼梯
    │   │   └── fibonacci.py    # 斐波那契数列
    │   ├── one_dimensional_dp/ # 一维动态规划
    │   │   ├── house_robber.py # LeetCode 198: 打家劫舍
    │   │   └── max_subarray.py # LeetCode 53: 最大子数组和
    │   └── two_dimensional_dp/ # 二维动态规划
    │       ├── unique_paths.py # LeetCode 62: 不同路径
    │       └── edit_distance.py # LeetCode 72: 编辑距离
    ├── 10_greedy_algorithms/  # 贪心算法
    │   ├── greedy_introduction.py # 贪心算法介绍
    │   ├── jump_game.py        # LeetCode 55: 跳跃游戏
    │   └── non_overlapping_intervals.py # LeetCode 435: 无重叠区间
    └── 11_advanced_data_structures/ # 高级数据结构
        ├── trie/              # 字典树(Trie)
        │   ├── trie_basics.py # Trie树基础
        │   └── implement_trie.py # LeetCode 208: 实现Trie(前缀树)
        ├── union_find/        # 并查集
        │   └── union_find_basics.py # 并查集基础与实现
        └── segment_tree/      # 线段树
            └── segment_tree_basics.py # 线段树基础与实现
```

## 零、算法与数据结构基础强化（2周）

### 第1周：算法基础
- **时间复杂度分析**
  - 大O概念讲解 (`00_algorithm_basics/time_complexity/big_o_concepts.py`)
  - 复杂度示例分析 (`00_algorithm_basics/time_complexity/complexity_examples.py`)
- **基本算法思想**
  - 递归基础 (`00_algorithm_basics/recursion/recursion_basics.py`)
  - 递归应用 (`00_algorithm_basics/recursion/recursion_applications.py`)
  - 迭代基础 (`00_algorithm_basics/iteration/iteration_basics.py`)
  - 分治法基础 (`00_algorithm_basics/divide_conquer/divide_conquer_basics.py`)
- 每日学习1个算法概念，通过Python代码实现并完成相关基础练习
- 推荐学习资源：
  - 《算法导论》相关章节
  - Coursera: Algorithms by Princeton University

### 第2周：数据结构基础
- **数组与列表**
  - 数组基础 (`01_data_structures/arrays/array_basics.py`)
  - 数组操作 (`01_data_structures/arrays/array_operations.py`)
- **链表**
  - 单链表实现 (`01_data_structures/linked_lists/singly_linked_list.py`)
  - 双链表实现 (`01_data_structures/linked_lists/doubly_linked_list.py`)
- **栈和队列**
  - 栈的实现 (`01_data_structures/stacks/stack_implementation.py`)
  - 队列的实现 (`01_data_structures/queues/queue_implementation.py`)
- **哈希表**
  - 哈希表实现 (`01_data_structures/hash_tables/hash_table_implementation.py`)
- **树**
  - 二叉树基础 (`01_data_structures/trees/binary_tree_basics.py`)
  - 二叉搜索树 (`01_data_structures/trees/binary_search_tree.py`)
  - 树的遍历 (`01_data_structures/trees/tree_traversal.py`)
- **图**
  - 图的表示 (`01_data_structures/graphs/graph_representation.py`)
  - 图的遍历 (`01_data_structures/graphs/graph_traversal.py`)
- 每日学习1个数据结构，编写完整实现并测试

## 一、基础阶段（4周）

### 第1周：数组与字符串
- 每天2-3道简单题，掌握基本操作
- 学习内容：数组遍历、字符串操作、双指针技巧
- 推荐题目：
  - 数组：1. Two Sum, 26. Remove Duplicates from Sorted Array
  - 字符串：14. Longest Common Prefix, 28. Find the Index of the First Occurrence in a String

### 第2周：链表与栈队列
- 每天2道题（简单为主，中等难度开始尝试）
- 学习内容：链表操作、栈和队列应用
- 推荐题目：
  - 链表：21. Merge Two Sorted Lists, 206. Reverse Linked List
  - 栈/队列：20. Valid Parentheses, 225. Implement Stack using Queues

### 第3周：哈希表与集合
- 每天2-3道题，增加中等难度题比例
- 学习内容：哈希表的应用、集合操作
- 推荐题目：49. Group Anagrams, 128. Longest Consecutive Sequence

### 第4周：树与递归
- 每天2道题，以中等难度为主
- 学习内容：二叉树遍历、递归解题思路
- 推荐题目：94. Binary Tree Inorder Traversal, 104. Maximum Depth of Binary Tree

## 二、进阶阶段（4周）

### 第5周：深度优先搜索与广度优先搜索
- 每天2道题，中等难度为主
- 学习内容：DFS、BFS在树和图中的应用
- 推荐题目：200. Number of Islands, 207. Course Schedule

### 第6周：动态规划入门
- 每天1-2道题，循序渐进
- 学习内容：DP基本概念、一维DP问题
- 推荐题目：70. Climbing Stairs, 198. House Robber, 121. Best Time to Buy and Sell Stock

### 第7周：二分查找与分治法
- 每天2道题，中等难度为主
- 学习内容：二分查找技巧、分治思想
- 推荐题目：33. Search in Rotated Sorted Array, 215. Kth Largest Element in an Array

### 第8周：贪心算法
- 每天2道题，难度混合
- 学习内容：贪心策略的应用场景
- 推荐题目：55. Jump Game, 435. Non-overlapping Intervals

## 三、高级阶段（4周）

### 第9周：复杂动态规划
- 每天1-2道题，中高难度
- 学习内容：二维DP、状态转移优化
- 推荐题目：62. Unique Paths, 72. Edit Distance, 300. Longest Increasing Subsequence

### 第10周：图论算法
- 每天1-2道题，中高难度
- 学习内容：图的表示、最短路径、拓扑排序
- 推荐题目：133. Clone Graph, 210. Course Schedule II, dijkstra算法应用

### 第11周：高级数据结构
- 每天1-2道题，难度混合
- 学习内容：Trie树、并查集、线段树
- 推荐题目：208. Implement Trie, 307. Range Sum Query - Mutable

### 第12周：综合练习与模拟面试
- 每天完成1道高难度题或2-3道混合难度题
- 进行模拟面试训练，限时解题
- 复习前面所学算法与数据结构

## 四、练习策略

1. **刷题节奏**：
   - 工作日每天至少1-2小时
   - 周末可以适当增加到3-4小时
   - 保持连续性，不要间断超过2天

2. **解题方法**：
   - 尝试独立思考30分钟
   - 如果没有思路，查看提示或解题思路
   - 理解解法后，关闭参考，独立编写Python代码
   - 优化解法，考虑时间和空间复杂度

3. **复习机制**：
   - 建立错题集，每周末复习一次
   - 相似题目归类整理，总结解题模板
   - 每月进行一次系统性回顾
   - 定期回顾Python实现文件，加深对知识点的理解

4. **基础知识强化**：
   - 每周拿出2-3小时专门用于复习算法与数据结构基础
   - 构建知识图谱，理清各算法和数据结构之间的关系
   - 对于不熟悉的概念，通过Python实现代码加深理解
   - 利用代码注释详细说明算法和数据结构的原理

5. **代码实现与教学**：
   - 每个知识点通过单独的Python文件实现，每个文件内容精简聚焦
   - 一个Python文件只包含一个知识点和最多两个相关LeetCode题目
   - 文件内部包含理论知识、实现代码和测试用例
   - 文件组织结构清晰，按主题和知识点分类存放
   - 采用渐进式教学：先讲解基本概念，再通过代码实现，最后通过题目应用
   - 通过实际案例和可视化输出增强理解

## 五、资源推荐

1. **学习网站**：
   - LeetCode官网及其讨论区
   - 算法可视化网站：VisuAlgo
   - 算法教程：GeeksforGeeks
   - MIT Open Courseware: Introduction to Algorithms
   - Stanford Algorithm Course
   - Python官方文档和教程

2. **书籍推荐**：
   - 《算法》(第4版) - Sedgewick & Wayne
   - 《算法导论》- MIT经典教材
   - 《Cracking the Coding Interview》- 面试宝典
   - 《Python算法教程》- Magnus Lie Hetland
   - 《Data Structures and Algorithms in Python》- Goodrich, Tamassia & Goldwasser

3. **进度追踪**：
   - 使用LeetCode的进度追踪功能
   - 创建自己的Github仓库记录解题代码和思路
   - 维护练习记录文件，详细记录每道题的解题思路和难点

祝你学习顺利！