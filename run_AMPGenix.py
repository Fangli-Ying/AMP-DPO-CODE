import os
import subprocess


def run_ampgenix():
    """直接调用命令行"""
    command = [
        'python', 'AMPGenix.py',
        '--device', '0',
        '--ntokens', '6-30',
        '--nsamples', '20',
        '--model_path', 'AMP_models/AmpGenix',
        '--prefix', 'W',#G, K, F, R, A, L, I, V, S and W
        '--topp', '5',
        '--temperature', '3',
        '--save_samples',
        '--save_samples_path', 'genix_generated/'
    ]

    # 确保目录存在
    os.makedirs('genix_generated/', exist_ok=True)

    # 执行命令
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("运行成功！")
        print(result.stdout)
    else:
        print("运行失败！")
        print(result.stderr)


if __name__ == "__main__":
    run_ampgenix()