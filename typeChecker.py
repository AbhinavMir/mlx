import os
import ast
from typing import Dict, List, Tuple, Optional

def analyze_file(file_path: str) -> Optional[Tuple[List[str], List[str]]]:
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        
        tree = ast.parse(content)
        missing_function_types = []
        missing_variable_types = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns is None:
                    missing_function_types.append(f"{node.name} (return)")
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self':
                        missing_function_types.append(f"{node.name} (parameter: {arg.arg})")
            
            elif isinstance(node, ast.AnnAssign):
                if node.annotation is None:
                    missing_variable_types.append(ast.unparse(node.target))
        
        return missing_function_types, missing_variable_types
    except SyntaxError:
        return None

def crawl_folder(folder_path: str) -> Dict[str, Tuple[List[str], List[str]]]:
    report = {}
    unparseable_files = []
    mlx_path = os.path.join(folder_path, 'python', 'mlx')
    
    if not os.path.exists(mlx_path):
        raise ValueError(f"The path {mlx_path} does not exist.")
    
    for root, _, files in os.walk(mlx_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                result = analyze_file(file_path)
                if result is None:
                    unparseable_files.append(file_path)
                else:
                    missing_functions, missing_variables = result
                    if missing_functions or missing_variables:
                        report[file_path] = (missing_functions, missing_variables)
    if unparseable_files:
        report['UNPARSEABLE_FILES'] = (unparseable_files, [])
    return report

def generate_markdown_report(report: Dict[str, Tuple[List[str], List[str]]]) -> str:
    output = "# Type Annotation Report\n\n"
    
    if 'UNPARSEABLE_FILES' in report:
        output += "## Files that couldn't be parsed due to syntax errors\n\n"
        for file in report['UNPARSEABLE_FILES'][0]:
            output += f"- {file}\n"
        output += "\n"
        del report['UNPARSEABLE_FILES']
    
    for file_path, (missing_functions, missing_variables) in report.items():
        output += f"## {file_path}\n\n"
        
        if missing_functions:
            output += "### Missing function type annotations\n\n"
            for func in missing_functions:
                output += f"- {func}\n"
            output += "\n"
        
        if missing_variables:
            output += "### Missing variable type annotations\n\n"
            for var in missing_variables:
                output += f"- {var}\n"
            output += "\n"
    
    return output

folder_path = '.' 
report = crawl_folder(folder_path)
report_text = generate_markdown_report(report)

with open('type_annotation_report.md', 'w') as f:
    f.write(report_text)

print("Report generated and saved as 'type_annotation_report.md'")