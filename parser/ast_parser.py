import ast


def _build_class_map(tree):
    class_map = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_map[id(child)] = node.name
    return class_map


def _extract_module_level(tree, source_code, file_path):
    chunks = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            code = ast.get_source_segment(source_code, node)
            names = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.append(elt.id)
            if names and code:
                chunks.append({
                    "function_name": f"[module] {', '.join(names)}",
                    "class_name": None,
                    "file_path": file_path,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "docstring": "",
                    "code": code,
                    "chunk_type": "module_variable"
                })

        elif isinstance(node, ast.AnnAssign):
            code = ast.get_source_segment(source_code, node)
            if isinstance(node.target, ast.Name) and code:
                chunks.append({
                    "function_name": f"[module] {node.target.id}",
                    "class_name": None,
                    "file_path": file_path,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "docstring": "",
                    "code": code,
                    "chunk_type": "module_variable"
                })

        elif isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node) or ""
            class_lines = source_code.splitlines()[node.lineno - 1:node.lineno + 5]
            chunks.append({
                "function_name": f"[class] {node.name}",
                "class_name": None,
                "file_path": file_path,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "docstring": docstring,
                "code": "\n".join(class_lines),
                "chunk_type": "class_definition"
            })

    return chunks


def extract_functions_from_file_content(source_code, file_path="unknown"):
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Skipping {file_path}: SyntaxError - {e}")
        return []

    class_map = _build_class_map(tree)
    chunks = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append({
                "function_name": node.name,
                "class_name": class_map.get(id(node)),
                "file_path": file_path,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "docstring": ast.get_docstring(node) or "",
                "code": ast.get_source_segment(source_code, node),
                "chunk_type": "function"
            })

    chunks.extend(_extract_module_level(tree, source_code, file_path))
    return chunks


def extract_functions_from_zip(zip_file):
    all_chunks = []

    for zip_path in zip_file.namelist():
        if not zip_path.endswith(".py"):
            continue
        with zip_file.open(zip_path) as f:
            try:
                source_code = f.read().decode("utf-8")
            except UnicodeDecodeError:
                print(f"Skipping {zip_path}: could not decode as UTF-8")
                continue
        all_chunks.extend(extract_functions_from_file_content(source_code, file_path=zip_path))

    functions = sum(1 for c in all_chunks if c["chunk_type"] == "function")
    variables = sum(1 for c in all_chunks if c["chunk_type"] == "module_variable")
    classes = sum(1 for c in all_chunks if c["chunk_type"] == "class_definition")
    print(f"Extracted: {functions} functions, {variables} module variables, {classes} classes")

    return all_chunks