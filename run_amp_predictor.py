#!/usr/bin/env python3
"""
AMP Sorter 预测脚本 - Python版本
包含默认参数，直接运行即可
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 默认参数配置
DEFAULT_CONFIG = {
    "batch_size": 200,
    "raw_data_path": r"Data\first.csv",
    "model_path": r"AMP_models\ProteoGPT",
    "classifier_path": r"AMP_models\AmpSorter\best_model.pt",
    "output_path": r"Data\Sequence_pred.csv",
    "candidate_amp_path": r"Data\Sequence_c_amps.csv"
}


def check_paths(config):
    """检查所有路径是否存在"""
    print("检查文件路径...")
    issues = []

    # 检查输入文件
    if not os.path.exists(config["raw_data_path"]):
        issues.append(f"输入文件不存在: {config['raw_data_path']}")

    # 检查模型目录
    if not os.path.exists(config["model_path"]):
        issues.append(f"模型路径不存在: {config['model_path']}")

    # 检查分类器文件
    if not os.path.exists(config["classifier_path"]):
        issues.append(f"分类器文件不存在: {config['classifier_path']}")

    # 检查输出目录是否存在，不存在则创建
    output_dir = os.path.dirname(config["output_path"])
    if output_dir and not os.path.exists(output_dir):
        print(f"创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    candidate_dir = os.path.dirname(config["candidate_amp_path"])
    if candidate_dir and not os.path.exists(candidate_dir):
        print(f"创建候选AMP目录: {candidate_dir}")
        os.makedirs(candidate_dir, exist_ok=True)

    # 显示检查结果
    if issues:
        print("⚠ 发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n是否继续? (y/n): ", end="")
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            return False
    else:
        print("✓ 所有路径检查通过")

    return True


def run_amp_prediction(config):
    """
    运行AMP预测
    """
    # 检查AMPSorter_predictor.py是否存在
    predictor_script = "AMPSorter_predictor.py"
    if not os.path.exists(predictor_script):
        # 尝试其他可能的脚本名
        possible_names = [
            "AMPSorter_predictor.py",
            "amp_predictor.py",
            "predictor.py",
            "main.py"
        ]
        for name in possible_names:
            if os.path.exists(name):
                predictor_script = name
                break
        else:
            print(f"错误: 找不到预测脚本")
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
        "--candidate_amp_path", config["candidate_amp_path"]
    ]

    print("\n" + "=" * 60)
    print("开始运行AMP预测")
    print("=" * 60)
    print(f"预测脚本: {predictor_script}")
    print(f"批次大小: {config['batch_size']}")
    print(f"输入文件: {config['raw_data_path']}")
    print(f"模型路径: {config['model_path']}")
    print(f"分类器: {config['classifier_path']}")
    print(f"输出文件: {config['output_path']}")
    print(f"候选AMP: {config['candidate_amp_path']}")
    print("=" * 60 + "\n")

    try:
        # 运行命令
        print("正在运行预测，请稍候...")
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True,
                                encoding='utf-8')

        # 打印输出
        if result.stdout:
            print("标准输出:")
            print(result.stdout)

        if result.stderr:
            print("标准错误:")
            print(result.stderr)

        print("\n" + "=" * 60)
        print("✓ 预测完成!")
        print("=" * 60)

        # 检查输出文件
        if os.path.exists(config["output_path"]):
            file_size = os.path.getsize(config["output_path"])
            print(f"主输出文件: {config['output_path']}")
            print(f"文件大小: {file_size:,} 字节")

            # 如果是CSV文件，尝试显示行数
            if config["output_path"].endswith('.csv'):
                with open(config["output_path"], 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"数据行数: {line_count}")

        if os.path.exists(config["candidate_amp_path"]):
            file_size = os.path.getsize(config["candidate_amp_path"])
            print(f"候选AMP文件: {config['candidate_amp_path']}")
            print(f"文件大小: {file_size:,} 字节")

        print("=" * 60)
        return True

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
        print(f"\n未知错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='AMP Sorter 预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                              # 使用默认参数运行
  %(prog)s --batch_size 100             # 修改批次大小
  %(prog)s --raw_data_path ./test.csv   # 指定其他输入文件
        """
    )

    # 添加参数，都设置为可选，有默认值
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG["batch_size"],
                        help=f'批次大小 (默认: {DEFAULT_CONFIG["batch_size"]})')

    parser.add_argument('--raw_data_path', type=str, default=DEFAULT_CONFIG["raw_data_path"],
                        help=f'输入序列文件路径 (默认: {DEFAULT_CONFIG["raw_data_path"]})')

    parser.add_argument('--model_path', type=str, default=DEFAULT_CONFIG["model_path"],
                        help=f'ProteoGPT模型路径 (默认: {DEFAULT_CONFIG["model_path"]})')

    parser.add_argument('--classifier_path', type=str, default=DEFAULT_CONFIG["classifier_path"],
                        help=f'AMP分类器模型路径 (默认: {DEFAULT_CONFIG["classifier_path"]})')

    parser.add_argument('--output_path', type=str, default=DEFAULT_CONFIG["output_path"],
                        help=f'预测结果输出路径 (默认: {DEFAULT_CONFIG["output_path"]})')

    parser.add_argument('--candidate_amp_path', type=str, default=DEFAULT_CONFIG["candidate_amp_path"],
                        help=f'候选AMP输出路径 (默认: {DEFAULT_CONFIG["candidate_amp_path"]})')

    # 添加一个快速测试模式
    parser.add_argument('--test', action='store_true',
                        help='测试模式，检查环境但不运行预测')

    # 解析参数
    args = parser.parse_args()

    # 创建配置字典
    config = {
        "batch_size": args.batch_size,
        "raw_data_path": args.raw_data_path,
        "model_path": args.model_path,
        "classifier_path": args.classifier_path,
        "output_path": args.output_path,
        "candidate_amp_path": args.candidate_amp_path
    }

    # 如果是测试模式
    if args.test:
        print("测试模式 - 只检查环境")
        print("=" * 60)
        check_paths(config)
        print("=" * 60)
        print("测试完成!")
        sys.exit(0)

    # 检查路径
    if not check_paths(config):
        print("路径检查失败，退出程序")
        sys.exit(1)

    # 运行预测
    success = run_amp_prediction(config)

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()