# MovieWhisper - 可解释电影推荐引擎

一个基于混合推荐算法的电影推荐系统，核心特点是提供可解释的推荐理由，告诉用户为什么推荐这部电影给你。

## 项目亮点

- 混合推荐策略：协同过滤 + 内容推荐加权融合
- 可解释性：每条推荐附带详细推荐理由和来源分析
- 交互式界面：基于 Streamlit 的 Web 界面，支持评分、推荐、用户画像

## 技术栈

- Python 3.10+
- pandas / numpy - 数据处理
- scikit-learn - 相似度计算
- Streamlit - Web 界面
- plotly - 可视化

## 快速开始

    pip install -r requirements.txt
    streamlit run app.py

## 项目结构

    src/
      data_loader.py     - 数据加载和预处理
      user_profile.py    - 用户画像构建
      collaborative.py   - 协同过滤算法
      content_based.py   - 内容推荐算法
      hybrid.py          - 混合推荐 + 可解释性
      explainer.py       - 推荐解释生成器
    app.py               - Streamlit 主入口
    tests/               - 单元测试
    notebooks/           - 数据探索笔记

## 推荐算法说明

### 协同过滤 (User-Based)
通过余弦相似度找到与目标用户口味相似的用户群体，推荐这些用户高分但目标用户尚未观看的电影。

### 内容推荐
基于电影类型标签 (genre) 构建特征向量，计算用户偏好向量与候选电影的相似度。

### 混合策略
两种推荐结果按 50/50 权重融合，综合得分排序后取 Top-10。

### 可解释性
每条推荐标注来源（相似用户推荐/风格匹配推荐/综合推荐），并给出具体理由文字。

## 数据来源

[MovieLens 100K](https://grouplens.org/datasets/movielens/) - GroupLens Research 项目提供的经典电影评分数据集。
