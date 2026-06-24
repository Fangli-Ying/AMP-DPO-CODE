#!/usr/bin/env python3
"""
BioToxiPept 预测脚本 - Python版本
包含默认参数，一键运行
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 默认参数配置
DEFAULT_CONFIG = {
    "batch_size": 200,
    "raw_data_path": r"Data\first_amp.csv",
    "model_path": r"AMP_models\ProteoGPT",
    "classifier_path": r"AMP_models\BioToxiPept\best_model.pt",
    "output_path": r"Data\Sequence_toxinpre.csv",
    "candidate_pep_path": r"Data\Sequence_nontoxin.csv"
}


def check_paths(config):
    """检查所有路径是否存在"""
    print("检查文件路径...")
    issues = []
    warnings = []

    # 检查输入文件
    if not os.path.exists(config["raw_data_path"]):
        issues.append(f" 输入文件不存在: {config['raw_data_path']}")

    # 检查模型目录
    if not os.path.exists(config["model_path"]):
        warnings.append(f" 模型路径不存在: {config['model_path']}")

    # 检查分类器文件
    if not os.path.exists(config["classifier_path"]):
        issues.append(f"分类器文件不存在: {config['classifier_path']}")

    # 检查输出目录是否存在，不存在则创建
    for path_key in ["output_path", "candidate_pep_path"]:
        path = config[path_key]
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            print(f"创建目录: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

    # 显示检查结果
    if warnings:
        print("\n警告信息:")
        for warning in warnings:
            print(f"  {warning}")

    if issues:
        print("\n发现以下严重问题:")
        for issue in issues:
            print(f"  {issue}")
        print("\n是否继续? (y/n): ", end="")
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            return False
    elif not warnings:
        print("所有路径检查通过")

    return True


def run_biotoxipept_prediction(config):
    """
    运行 BioToxiPept 预测
    """
    # 检查 BioToxiPept.py 是否存在
    predictor_script = "BioToxiPept.py"
    if not os.path.exists(predictor_script):
        # 尝试其他可能的脚本名
        possible_names = [
            "BioToxiPept.py",
            "biotoxipept.py",
            "BioToxiPept_predictor.py",
            "toxin_predictor.py",
            "main.py"
        ]
        for name in possible_names:
            if os.path.exists(name):
                predictor_script = name
                print(f"找到预测脚本: {name}")
                break
        else:
            print(f"错误: 找不到 BioToxiPept 预测脚本")
            print("请在当前目录放置以下文件之一:")
            for name in possible_names:
                print(f"  - {name}")
            return False

    # 构建命令参数
    cmd = [
        "python",
        predictor_script,
        "--batch_size", str(config["batch_size"]),
        "--raw_data_path", config["raw_data_path"],
        "--model_path", config["model_path"],
        "--classifier_path", config["classifier_path"],
        "--output_path", config["output_path"],
        "--candidate_pep_path", config["candidate_pep_path"]
    ]

    # 显示运行信息
    print("\n" + "=" * 70)
    print("🧬 BioToxiPept 毒性肽预测")
    print("=" * 70)
    print(f"预测脚本: {predictor_script}")
    print(f"批次大小: {config['batch_size']}")
    print(f"输入文件: {config['raw_data_path']}")
    print(f"基础模型: {config['model_path']}")
    print(f"分类器: {config['classifier_path']}")
    print(f"毒性肽输出: {config['output_path']}")
    print(f"非毒性肽输出: {config['candidate_pep_path']}")
    print("=" * 70 + "\n")

    try:
        # 运行命令
        print("⏳ 正在运行毒性肽预测，请稍候...")
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True,
                                encoding='utf-8',
                                timeout=3600)  # 1小时超时

        # 显示输出信息
        if result.stdout:
            print("标准输出:")
            print("-" * 40)
            print(result.stdout.strip())
            print("-" * 40)

        if result.stderr and result.stderr.strip():
            print(" 标准错误:")
            print("-" * 40)
            print(result.stderr.strip())
            print("-" * 40)

        print("\n" + "=" * 70)
        print("BioToxiPept 预测完成!")
        print("=" * 70)

        # 检查输出文件
        output_files = [
            ("毒性肽预测结果", config["output_path"]),
            ("非毒性肽结果", config["candidate_pep_path"])
        ]

        for file_desc, file_path in output_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"\n {file_desc}:")
                print(f"  文件路径: {file_path}")
                print(f"  文件大小: {file_size:,} 字节")

                # 如果是CSV文件，显示行数
                if file_path.endswith('.csv'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if lines:
                                # 第一行可能是标题
                                data_lines = len(lines) - (1 if lines[0].strip().startswith('seq') else 0)
                                print(f"  数据行数: {data_lines}")
                                print(f"  标题行: {'有' if len(lines) > data_lines else '无'}")
                    except Exception as e:
                        print(f"  无法读取文件详细信息: {e}")
            else:
                print(f"\n {file_desc}未生成: {file_path}")

        print("\n" + "=" * 70)
        return True

    except subprocess.TimeoutExpired:
        print("\n⏰ 错误: 预测超时（超过1小时）")
        print("可能是数据量太大或模型加载缓慢")
        return False

    except subprocess.CalledProcessError as e:
        print("\n错误: 预测脚本执行失败")
        print(f"退出码: {e.returncode}")
        print(f"错误信息:\n{e.stderr}")
        return False

    except FileNotFoundError:
        print("\n错误: 找不到 'python' 命令")
        print("请确保Python已正确安装并添加到系统PATH")
        return False

    except Exception as e:
        print(f"\n 未知错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_config_file():
    """创建配置文件"""
    config_content = """# BioToxiPept 配置文件
# 可以在此修改默认路径，然后运行时不带参数即可使用

batch_size = 200
raw_data_path = D:\\AMP_Project-main\\AMP_Project-main\\Data\\Sequence.csv
model_path = D:\\AMP_Project-main\\AMP_Project-main\\AMP_models\\ProteoGPT
classifier_path = D:\\AMP_Project-main\\AMP_Project-main\\AMP_models\\BioToxiPept\\best_model.pt
output_path = D:\\AMP_Project-main\\AMP_Project-main\\Data\\Sequence_toxinpre.csv
candidate_pep_path = D:\\AMP_Project-main\\AMP_Project-main\\Data\\Sequence_nontoxin.csv
"""

    with open("biotoxipept_config.ini", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("配置文件已创建: biotoxipept_config.ini")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='BioToxiPept 毒性肽预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_biotoxipept.py                    # 使用默认参数运行
  python run_biotoxipept.py --batch_size 100   # 修改批次大小
  python run_biotoxipept.py --test             # 测试模式
  python run_biotoxipept.py --config           # 生成配置文件
  python run_biotoxipept.py --help             # 显示帮助

默认参数:
  批次大小: 200
  输入文件: D:\\AMP_Project-main\\AMP_Project-main\\Data\\Sequence.csv
  模型路径: D:\\AMP_Project-main\\AMP_Project-main\\AMP_models\\ProteoGPT
  分类器: D:\\AMP_Project-main\\AMP_Project-main\\AMP_models\\BioToxiPept\\best_model.pt
  毒性肽输出: D:\\AMP_Project-main\\AMP_Project-main\\Data\\Sequence_toxinpre.csv
  非毒性肽输出: D:\\AMP_Project-main\\AMP_Project-main\\Data\\Sequence_nontoxin.csv
        """
    )

    # 添加参数，都设置为可选，有默认值
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG["batch_size"],
                        help=f'批次大小 (默认: {DEFAULT_CONFIG["batch_size"]})')

    parser.add_argument('--raw_data_path', type=str, default=DEFAULT_CONFIG["raw_data_path"],
                        help=f'输入序列文件路径 (默认: {DEFAULT_CONFIG["raw_data_path"]})')

    parser.add_argument('--model_path', type=str, default=DEFAULT_CONFIG["model_path"],
                        help=f'基础模型路径 (默认: {DEFAULT_CONFIG["model_path"]})')

    parser.add_argument('--classifier_path', type=str, default=DEFAULT_CONFIG["classifier_path"],
                        help=f'毒性分类器模型路径 (默认: {DEFAULT_CONFIG["classifier_path"]})')

    parser.add_argument('--output_path', type=str, default=DEFAULT_CONFIG["output_path"],
                        help=f'毒性肽预测结果输出路径 (默认: {DEFAULT_CONFIG["output_path"]})')

    parser.add_argument('--candidate_pep_path', type=str, default=DEFAULT_CONFIG["candidate_pep_path"],
                        help=f'非毒性肽输出路径 (默认: {DEFAULT_CONFIG["candidate_pep_path"]})')

    # 添加功能选项
    parser.add_argument('--test', action='store_true',
                        help='测试模式，只检查环境不运行预测')

    parser.add_argument('--config', action='store_true',
                        help='生成配置文件，然后退出')

    # 解析参数
    args = parser.parse_args()

    # 如果是生成配置文件模式
    if args.config:
        create_config_file()
        sys.exit(0)

    # 创建配置字典
    config = {
        "batch_size": args.batch_size,
        "raw_data_path": args.raw_data_path,
        "model_path": args.model_path,
        "classifier_path": args.classifier_path,
        "output_path": args.output_path,
        "candidate_pep_path": args.candidate_pep_path
    }

    # 显示配置信息
    print("\n" + "=" * 70)
    print("BioToxiPept 配置信息")
    print("=" * 70)
    for key, value in config.items():
        print(f"{key:20}: {value}")
    print("=" * 70)

    # 如果是测试模式
    if args.test:
        print("\n测试模式 - 只检查环境")
        if check_paths(config):
            print("\n环境检查通过，可以正常运行预测")
        else:
            print("\n环境检查失败，请解决上述问题")
        sys.exit(0)

    # 检查路径
    if not check_paths(config):
        print("路径检查失败，退出程序")
        sys.exit(1)

    # 运行预测
    success = run_biotoxipept_prediction(config)

    # 返回退出码
    if success:
        print("\nBioToxiPept 预测任务成功完成!")
    else:
        print("\n BioToxiPept 预测任务失败!")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()