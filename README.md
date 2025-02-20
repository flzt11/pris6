# pris7
AI课程任务7 - 基于 KMeans 手写字符识别的实现，提供了数据集加载、模型训练、测试以及可视化功能。


## 使用说明
1. 创建并激活虚拟环境：
    ```bash
    conda create -n pris6 python=3.6
    conda activate pris6
    ```

2. 克隆项目并安装依赖：
    ```bash
    git clone https://github.com/flzt11/pris6.git 
    pip install ./pris7
    ```

## 启动说明
1. 导入并启动训练和测试：
    ```python
    import pris6
    pris6.cv("load_dataset")  # 读取数据
    pris6.cv("run")  # 运行
    pris6.cv("visualize")   # 可视化
    ```

## 依赖项
- Python 3.7
- Conda
- PyTorch (CUDA 11.7)

