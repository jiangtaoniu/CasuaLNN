import ast
import os
import sys
import importlib.util

def check_syntax_and_imports():
    root_dir = '.'
    bad_files = []
    imports_map = {}

    for root, dirs, files in os.walk(root_dir):
        if '.claude' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        source = f.read()

                    tree = ast.parse(source)
                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)
                            elif node.level > 0:
                                # Relative import
                                imports.append(f"RELATIVE_{node.level}")

                    imports_map[filepath] = imports
                except SyntaxError as e:
                    print(f"Syntax error in {filepath}: {e}")
                    bad_files.append(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    print(f"Analyzed {len(imports_map)} files.")
    return imports_map, bad_files

if __name__ == "__main__":
    imports_map, bad_files = check_syntax_and_imports()
    root_dir = '.'
    if bad_files:
        sys.exit(1)
    for fp, imps in imports_map.items():
        local_imps = [imp for imp in imps if any(imp.startswith(prefix) for prefix in ['exp', 'models', 'layers', 'data_provider', 'utils', 'ncps'])]
        for imp in local_imps:
            # try to resolve importing internal packages
            parts = imp.split('.')
            if parts[0] in ['exp', 'models', 'layers', 'data_provider', 'utils', 'ncps']:
                mod_path = os.path.join(root_dir, *parts) + '.py'
                dir_path = os.path.join(root_dir, *parts, '__init__.py')
                if not os.path.exists(mod_path) and not os.path.exists(dir_path):
                    print(f"BROKEN IMPORT in {fp}: {imp}")
