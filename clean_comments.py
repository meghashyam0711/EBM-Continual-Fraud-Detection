"""
clean_comments.py - Remove non-docstring comments from Python source files in the project.
This script parses .py files, preserving docstrings (triple-quoted strings) and removing any
standalone line comments (starting with '#') as well as inline comments after code.
It also cleans CSS and JS files by stripping block comments (/* ... */) and line comments
('// ...').

Usage:
    python clean_comments.py
"""
import os
import re
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent

def remove_comments_from_line(line: str, in_multiline_quote=None) -> tuple[str, str | None]:
    i = 0
    n = len(line)
    in_single_quote = False
    in_double_quote = False
    escaped = False
    
    while i < n:
        char = line[i]
        
        if escaped:
            escaped = False
            i += 1
            continue
            
        if char == '\\':
            escaped = True
            i += 1
            continue
            
        if in_multiline_quote:
            q_type = in_multiline_quote
            if line[i:i+3] == q_type:
                in_multiline_quote = None
                i += 3
            else:
                i += 1
            continue
            
        if not in_single_quote and not in_double_quote:
            if line[i:i+3] == '"""':
                in_multiline_quote = '"""'
                i += 3
                continue
            if line[i:i+3] == "'''":
                in_multiline_quote = "'''"
                i += 3
                continue
            if char == "'":
                in_single_quote = True
                i += 1
                continue
            if char == '"':
                in_double_quote = True
                i += 1
                continue
            if char == '#':
                cleaned = line[:i].rstrip()
                if line.endswith('\n'):
                    cleaned += '\n'
                return cleaned, in_multiline_quote
        else:
            if in_single_quote and char == "'":
                in_single_quote = False
            elif in_double_quote and char == '"':
                in_double_quote = False
        i += 1
        
    return line, in_multiline_quote

def clean_python_file(path: pathlib.Path):
    if path.name == 'clean_comments.py':
        return
    with path.open('r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_multiline_quote = None
    for line in lines:
        cleaned_line, in_multiline_quote = remove_comments_from_line(line, in_multiline_quote)
        original_stripped = line.strip()
        cleaned_stripped = cleaned_line.strip()
        
        if original_stripped.startswith('#') and not cleaned_stripped:
            continue
            
        new_lines.append(cleaned_line)
        
    with path.open('w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Cleaned {path}")

def clean_css_js_file(path: pathlib.Path):
    with path.open('r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    if path.suffix == '.js':
        content = re.sub(r'(?<!:)\/\/.*', '', content)
    with path.open('w', encoding='utf-8') as f:
        f.write(content)
    print(f"Cleaned {path}")

def main():
    for root, _, files in os.walk(PROJECT_ROOT):
        for fname in files:
            fpath = pathlib.Path(root) / fname
            if fpath.suffix == '.py':
                clean_python_file(fpath)
            elif fpath.suffix in {'.css', '.js'}:
                clean_css_js_file(fpath)

if __name__ == '__main__':
    main()

