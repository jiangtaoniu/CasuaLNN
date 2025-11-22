# Filename: z_getModelCode.py
# (Originally named create_full_summary.py)
#
# This script should be placed in the 'models' folder to run correctly.

import os
import ast
import re
from typing import List, Set, Dict, Tuple

"""
This script is a developer utility for creating a self-contained code summary
of a selected model and all its layer dependencies. It analyzes the model file,
finds all direct and recursive dependencies within the 'layers' directory, and then
consolidates the source code of the selected model(s) and all dependencies into
a single, easy-to-share text file.
"""

# --- Core Helper Functions ---

def find_project_root(start_path: str, markers: List[str]) -> str:
    """
    Traverses up from a starting path to find the project root directory.

    The root is identified by the presence of marker files/directories like '.git'
    or by being the parent of both 'layers' and 'models' directories.

    Args:
        start_path (str): The initial path to start the search from (e.g., the script's directory).
        markers (List[str]): A list of marker file/directory names that indicate the project root.

    Returns:
        str: The absolute path to the project root.

    Raises:
        FileNotFoundError: If the project root cannot be determined.
    """
    current_path = os.path.abspath(start_path)
    while True:
        # Check if the current directory contains both 'layers' and 'models' subdirectories
        if os.path.isdir(os.path.join(current_path, 'layers')) and \
                os.path.isdir(os.path.join(current_path, 'models')):
            return current_path

        # Check for presence of root markers
        for marker in markers:
            if os.path.exists(os.path.join(current_path, marker)):
                return current_path

        parent_path = os.path.dirname(current_path)
        if parent_path == current_path: # Reached the filesystem root
            raise FileNotFoundError("Could not find the project root. Please ensure the script is "
                                    "run from the 'models' folder and that a 'layers' folder exists "
                                    "at the same project level.")
        current_path = parent_path


def get_files_in_current_directory(scan_path: str, excluded_files: Set[str]) -> List[str]:
    """
    Non-recursively gets a list of Python files in the current directory for user selection.

    Args:
        scan_path (str): The directory path to scan.
        excluded_files (Set[str]): A set of filenames to exclude from the list.

    Returns:
        List[str]: A sorted list of selectable Python filenames.
    """
    file_list = [
        f for f in os.listdir(scan_path)
        if f.endswith('.py') and os.path.isfile(os.path.join(scan_path, f)) and f not in excluded_files
    ]
    return sorted(file_list)


def find_layer_imports_in_models(model_files: List[str], models_dir: str) -> Set[str]:
    """
    Parses the selected model files to find all directly imported modules from the 'layers' package.

    Args:
        model_files (List[str]): List of selected model filenames.
        models_dir (str): The absolute path to the 'models' directory.

    Returns:
        Set[str]: A set of unique layer module names that were directly imported.
    """
    layer_modules = set()
    # Regex to match lines like: from layers.some_module import ...
    # It captures the module name 'some_module'.
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


def find_all_recursive_dependencies(project_root: str, initial_layer_modules: Set[str]) -> Dict[str, str]:
    """
    Recursively finds all nested dependencies starting from an initial set of layer modules.

    Args:
        project_root (str): The absolute path to the project root.
        initial_layer_modules (Set[str]): The initial set of layer module names to start the search from.

    Returns:
        Dict[str, str]: A dictionary mapping each found module name to its relative path from the project root.
    """
    modules_to_process = set(initial_layer_modules)
    processed_modules = set()
    found_module_paths: Dict[str, str] = {}  # {module_name: relative_path_to_root}

    layers_dir = os.path.join(project_root, 'layers')
    if not os.path.isdir(layers_dir):
        print(f"[!] Error: 'layers' directory not found at '{layers_dir}'.")
        return {}

    # Cache all Python files in the layers directory for quick lookups
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
            relative_path = os.path.relpath(full_path, project_root).replace("\", "/")
            found_module_paths[module_name] = relative_path
            print(f"  -> Including dependency '{module_name}' from: {relative_path}")

            # Parse the current file for further 'from layers...' imports
            import_pattern = re.compile(r"^\s*from\s+layers\.([a-zA-Z0-9_]+)", re.MULTILINE)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    matches = import_pattern.findall(content)
                    for new_dependency in matches:
                        if new_dependency not in processed_modules:
                            modules_to_process.add(new_dependency)
            except Exception:
                pass  # Ignore files that can't be read
        else:
            print(f"  - Warning: Layer module '{module_name}' was imported but not found in the 'layers' directory.")

    print(f"[*] Total of {len(found_module_paths)} layer modules will be included.")
    return found_module_paths


def create_final_summary(project_root: str, selected_models: List[str],
                         all_layer_paths: Dict[str, str], output_filename: str):
    """
    Generates the final code summary file by concatenating the content of all
    selected model and layer files.

    Args:
        project_root (str): The absolute path to the project root.
        selected_models (List[str]): List of selected model filenames.
        all_layer_paths (Dict[str, str]): Dictionary of all dependent layer modules and their paths.
        output_filename (str): The name of the output summary file.
    """
    models_dir = os.path.join(project_root, 'models')
    output_path = os.path.join(models_dir, output_filename)

    print("\n[Analysis Step 3/3] Creating final code summary...")

    with open(output_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(f"# Code Summary for Selected Models and All Their Dependencies\n")
        summary_file.write(f"# Project Root: {project_root}\n\n")

        # --- Write a manifest of all included modules ---
        summary_file.write("Summary includes the following modules:\n\n")

        summary_file.write("# --- Layer Modules ---")
        sorted_layers = sorted(all_layer_paths.items())
        for module_name, path in sorted_layers:
            summary_file.write(f"- {module_name} ({path})\n")

        summary_file.write("\n# --- Model Modules ---")
        for model_file in selected_models:
            summary_file.write(f"- {model_file} (models/{model_file})\n")

        summary_file.write("\n" + "=" * 80 + "\n\n")

        # --- Concatenate the content of all files ---
        all_files_to_write = []
        # Add all layer files
        for _, path in sorted_layers:
            all_files_to_write.append({'path': path})
        # Add all model files
        for model_file in sorted(selected_models):
            all_files_to_write.append({'path': f"models/{model_file}"})

        total_files = len(all_files_to_write)
        for i, file_info in enumerate(all_files_to_write):
            relative_path = file_info['path']
            module_name = os.path.splitext(os.path.basename(relative_path))[0]
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


# --- Main Execution Block ---

if __name__ == "__main__":
    OUTPUT_FILENAME = "z_CodeSummary_Full_Stack.txt"

    try:
        # Step 1: Determine project and model directory paths
        models_dir = os.path.dirname(os.path.abspath(__file__))
        project_markers = ['.git', 'pyproject.toml']
        project_root = find_project_root(models_dir, project_markers)

        print("=" * 60)
        print("    Integrated Model & Layer Code Summarizer")
        print("=" * 60)
        print(f"[*] Project Root detected: {project_root}")
        print(f"[*] Running in 'models' directory: {models_dir}")

        # Step 2: List selectable model files and get user input
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
                    # Prompt user for selection
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

            # Step 3 & 4: Perform dependency analysis
            direct_layer_deps = find_layer_imports_in_models(selected_models, models_dir)
            all_layer_dependencies = find_all_recursive_dependencies(project_root, direct_layer_deps)

            # Step 5: Generate the final summary file
            if not all_layer_dependencies and not selected_models:
                print("\n[*] No modules to summarize. Exiting.")
            else:
                create_final_summary(project_root, selected_models, all_layer_dependencies, OUTPUT_FILENAME)

    except Exception as e:
        print(f"\n[!!!] An unexpected error occurred: {e}")

    print("\n[*] Process finished.")
