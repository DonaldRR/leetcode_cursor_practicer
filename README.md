# LeetCode学习项目

这是一个系统性学习算法和数据结构并练习LeetCode题目的项目。

## 项目结构

- PLAN.md - 详细学习计划
- PRACTICE_LOG.md - 练习记录
- code_examples/ - 代码示例和LeetCode解题

## 环境设置

本项目使用Python虚拟环境管理依赖，需要Python 3.8+。

### 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 必要依赖

- colorama - 终端彩色文本输出
- matplotlib - 可视化图表生成
- networkx - 图形网络分析与可视化
- pydot - 图形渲染工具

## 使用方法

1. 激活虚拟环境后，可以运行代码示例：

```bash
python code_examples/00_algorithm_basics/recursion/visualization/climbing_stairs_detailed.py
```

2. 学习进度跟踪：查看PRACTICE_LOG.md记录您的练习历程。

3. 学习计划：查看PLAN.md了解后续学习内容。

## 注意事项

- 虚拟环境目录(venv/)已被添加到.gitignore，不会被提交到版本控制
- 可视化生成的图片文件也不会被提交
- 如需在新机器上设置，只需克隆仓库后按上述步骤创建虚拟环境并安装依赖
