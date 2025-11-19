"""
验证保存的模型参数是否正确
"""
import paddle
import os
from DQN import Agent, build_model
from GameInterface import GameInterface

def verify_model():
    """验证模型文件的完整性"""

    model_path = "final.pdparams"

    print("=" * 60)
    print("验证模型参数文件")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件 {model_path} 不存在")
        return False

    file_size = os.path.getsize(model_path) / 1024  # KB
    print(f"✅ 模型文件存在: {model_path}")
    print(f"   文件大小: {file_size:.2f} KB")

    # 尝试加载模型
    try:
        # 初始化环境和agent
        feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
        feature_map_width = GameInterface.FEATURE_MAP_WIDTH
        action_dim = GameInterface.ACTION_NUM
        feature_dim = feature_map_height * feature_map_width * 2

        agent = Agent(build_model, feature_dim, action_dim, e_greed=0.1, e_greed_decrement=0)

        # 加载参数
        state_dict = paddle.load(model_path)
        agent.policy_net.set_state_dict(state_dict)

        print("✅ 模型参数加载成功")
        print(f"   参数数量: {len(state_dict)} 个张量")

        # 显示参数形状
        print("\n📊 模型参数详情:")
        total_params = 0
        for name, param in state_dict.items():
            shape = param.shape
            num_params = 1
            for dim in shape:
                num_params *= dim
            total_params += num_params
            print(f"   {name}: {shape} ({num_params:,} 参数)")

        print(f"\n   总参数量: {total_params:,}")

        # 测试前向传播
        import numpy as np
        test_feature = np.random.randn(1, feature_dim).astype('float32')
        action = agent.predict(test_feature)
        print(f"\n✅ 前向传播测试成功")
        print(f"   输入形状: {test_feature.shape}")
        print(f"   输出动作: {action}")

        print("\n" + "=" * 60)
        print("✅ 模型验证通过！模型文件完整且可用。")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 错误：加载模型失败")
        print(f"   错误信息: {e}")
        return False

if __name__ == "__main__":
    verify_model()
