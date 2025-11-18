"""
测试视频录制 - 只录制最高分局
"""
from record_top_games import *

if __name__ == "__main__":
    print("测试录制最高分局视频...")

    # 我们已经知道Seed 34是最高分
    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH
    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2

    env = GameInterface()
    agent = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
    agent.policy_net.set_state_dict(paddle.load("final.pdparams"))

    # 创建videos目录
    if not os.path.exists('videos'):
        os.makedirs('videos')

    # 录制最高分局
    result = record_game_video(
        agent, env,
        seed=34,
        output_path='videos/best_game_seed34.mp4',
        fps=15,  # 15 FPS
        show_q_values=True
    )

    if result:
        print(f"\n✅ 测试成功!")
        print(f"视频文件: {result['file']}")
        print(f"文件大小: {result['size_mb']:.2f} MB")
