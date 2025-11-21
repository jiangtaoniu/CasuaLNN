# 文件名: create_full_summary.py
# 请将此文件放置在您的 models 文件夹下运行

import os
import ast
import re
from typing import List, Set, Dict, Tuple


# --- 核心辅助函数 ---

def find_project_root(start_path: str, markers: List[str]) -> str:
    """从起始路径向上回溯，寻找项目根目录。"""
    current_path = os.path.abspath(start_path)
    while True:
        # 检查当前目录是否为 'layers' 或 'models' 的父目录
        if os.path.isdir(os.path.join(current_path, 'layers')) and \
                os.path.isdir(os.path.join(current_path, 'models')):
            return current_path

        for marker in markers:
            if os.path.exists(os.path.join(current_path, marker)):
                return current_path

        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            raise FileNotFoundError("无法找到项目根目录。请确保脚本位于 'models' 文件夹内，"
                                    "且同级目录存在 'layers' 文件夹。")
        current_path = parent_path


def get_files_in_current_directory(scan_path: str, excluded_files: Set[str]) -> List[str]:
    """非递归地获取当前目录下的Python文件供用户选择。"""
    file_list = [
        f for f in os.listdir(scan_path)
        if f.endswith('.py') and os.path.isfile(os.path.join(scan_path, f)) and f not in excluded_files
    ]
    return sorted(file_list)


def find_layer_imports_in_models(model_files: List[str], models_dir: str) -> Set[str]:
    """
    步骤2: 从选定的模型文件中解析出直接导入的 'layers' 模块名。
    """
    layer_modules = set()
    # 正则表达式匹配 'from layers.some_module import ...'
    # 它会捕获 'some_module'
    import_pattern = re.compile(r"^\s*from\s+layers\.([a-zA-Z0-9_]+)", re.MULTILINE)

    print("\n[Analysis Step 1/3] Finding direct layer dependencies in selected models...")
    for model_file in model_files:
        full_path = os.path.join(models_dir, model_file)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = import_pattern.findall(content)
                for module_name in matches:
                    layer_modules.add(module_name)
                    print(f"  - Found direct import in '{model_file}': layers.{module_name}")
        except Exception as e:
            print(f"  - Warning: Could not parse imports in {model_file}. Reason: {e}")

    print(f"[*] Found {len(layer_modules)} unique direct layer dependencies.")
    return layer_modules


def find_all_recursive_dependencies(
        project_root: str,
        initial_layer_modules: Set[str]
) -> Dict[str, str]:
    """
    步骤3: 从初始'layers'模块开始，递归查找所有相关的'layers'模块。
    """
    modules_to_process = set(initial_layer_modules)
    processed_modules = set()
    found_module_paths: Dict[str, str] = {}  # {module_name: relative_path_to_root}

    layers_dir = os.path.join(project_root, 'layers')
    if not os.path.isdir(layers_dir):
        print(f"[!] Error: 'layers' directory not found at '{layers_dir}'.")
        return {}

    # 缓存layers目录下的所有py文件
    layer_module_cache: Dict[str, str] = {
        os.path.splitext(f)[0]: os.path.join(layers_dir, f)
        for f in os.listdir(layers_dir) if f.endswith('.py')
    }

    print("\n[Analysis Step 2/3] Recursively finding all nested layer dependencies...")
    while modules_to_process:
        module_name = modules_to_process.pop()
        if module_name in processed_modules:
            continue

        processed_modules.add(module_name)

        if module_name in layer_module_cache:
            full_path = layer_module_cache[module_name]
            relative_path = os.path.relpath(full_path, project_root).replace("\\", "/")
            found_module_paths[module_name] = relative_path
            print(f"  -> Including dependency '{module_name}' from: {relative_path}")

            # 解析当前文件的 'from layers...' 导入
            import_pattern = re.compile(r"^\s*from\s+layers\.([a-zA-Z0-9_]+)", re.MULTILINE)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    matches = import_pattern.findall(content)
                    for new_dependency in matches:
                        if new_dependency not in processed_modules:
                            modules_to_process.add(new_dependency)
            except Exception:
                pass
        else:
            print(f"  - Warning: Layer module '{module_name}' was imported but not found in 'layers' directory.")

    print(f"[*] Total of {len(found_module_paths)} layer modules will be included.")
    return found_module_paths


def create_final_summary(
        project_root: str,
        selected_models: List[str],
        all_layer_paths: Dict[str, str],
        output_filename: str
):
    """
    步骤4: 生成最终的代码摘要文件。
    """
    models_dir = os.path.join(project_root, 'models')
    output_path = os.path.join(models_dir, output_filename)

    print("\n[Analysis Step 3/3] Creating final code summary...")

    with open(output_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(f"# Code Summary for Selected Models and All Their Dependencies\n")
        summary_file.write(f"# Project Root: {project_root}\n\n")

        # --- 模块清单 ---
        summary_file.write("Summary includes the following modules:\n\n")

        summary_file.write("# --- Layer Modules ---\n")
        sorted_layers = sorted(all_layer_paths.items())
        for module_name, path in sorted_layers:
            summary_file.write(f"- {module_name} ({path})\n")

        summary_file.write("\n# --- Model Modules ---\n")
        for model_file in selected_models:
            summary_file.write(f"- {model_file} (models/{model_file})\n")

        summary_file.write("\n" + "=" * 80 + "\n\n")

        # --- 写入代码内容 ---
        all_files_to_write = []
        # 添加所有layer文件
        for module, path in sorted_layers:
            all_files_to_write.append({'name': module, 'path': path})
        # 添加所有model文件
        for model_file in sorted(selected_models):
            all_files_to_write.append({'name': os.path.splitext(model_file)[0], 'path': f"models/{model_file}"})

        total_files = len(all_files_to_write)
        for i, file_info in enumerate(all_files_to_write):
            module_name = file_info['name']
            relative_path = file_info['path']
            full_path = os.path.join(project_root, relative_path)

            print(f"  [{i + 1}/{total_files}] Adding content of: {relative_path}")

            summary_file.write("=" * 80 + "\n")
            summary_file.write(f"## MODULE: {module_name}\n")
            summary_file.write(f"#  PATH: {relative_path}\n")
            summary_file.write("=" * 80 + "\n\n")

            try:
                with open(full_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    summary_file.write(content)
            except Exception as e:
                error_msg = f"# !!! ERROR: Could not read file '{relative_path}'. Reason: {e} !!!\n"
                summary_file.write(error_msg)
                print(f"    -> ERROR reading file: {e}")
            summary_file.write("\n\n\n")

    print(f"\n[*] Summary successfully created at: {output_path}")


# --- 主执行函数 ---

if __name__ == "__main__":
    OUTPUT_FILENAME = "z_CodeSummary_Full_Stack.txt"

    try:
        # 步骤 1: 确定路径
        models_dir = os.path.dirname(os.path.abspath(__file__))
        project_markers = ['.git', 'pyproject.toml']
        project_root = find_project_root(models_dir, project_markers)

        print("=" * 60)
        print("    Integrated Model & Layer Code Summarizer")
        print("=" * 60)
        print(f"[*] Project Root detected: {project_root}")
        print(f"[*] Running in 'models' directory: {models_dir}")

        # 步骤 2: 列出并选择模型文件
        excluded_files = {OUTPUT_FILENAME, os.path.basename(__file__)}
        selectable_models = get_files_in_current_directory(models_dir, excluded_files)

        if not selectable_models:
            print("\n[!] No Python models found in the current directory. Exiting.")
        else:
            print("\nPlease select the model file(s) you want to summarize:")
            for i, model_name in enumerate(selectable_models):
                print(f"  [{i + 1}] {model_name}")

            selected_models = []
            while True:
                try:
                    raw_input = input(">>> Enter file numbers, separated by spaces (e.g., 1 3): ")
                    if not raw_input.strip(): continue

                    selected_indices = {int(i) - 1 for i in raw_input.split()}
                    valid_indices = {idx for idx in selected_indices if 0 <= idx < len(selectable_models)}

                    if not valid_indices:
                        print("[!] No valid selection. Please try again.")
                        continue

                    selected_models = [selectable_models[i] for i in sorted(list(valid_indices))]
                    break
                except ValueError:
                    print("[!] Invalid input. Please enter numbers only.")

            print("\nSelected models for analysis:")
            for model in selected_models:
                print(f"  - {model}")

            # 步骤 3 & 4: 执行两阶段依赖分析
            direct_layer_deps = find_layer_imports_in_models(selected_models, models_dir)
            all_layer_dependencies = find_all_recursive_dependencies(project_root, direct_layer_deps)

            # 步骤 5: 生成最终摘要
            if not all_layer_dependencies and not selected_models:
                print("\n[*] No modules to summarize. Exiting.")
            else:
                create_final_summary(project_root, selected_models, all_layer_dependencies, OUTPUT_FILENAME)

    except Exception as e:
        print(f"\n[!!!] An unexpected error occurred: {e}")

    print("\n[*] Process finished.")