import esprima


def _get_source_segment(source_lines, start_line, end_line):
    """Extract source lines (1-indexed)."""
    return "\n".join(source_lines[start_line - 1:end_line])


def _extract_function_name(node):
    """Get function name from various node types."""
    if node.type == "FunctionDeclaration":
        return node.id.name if node.id else "<anonymous>"
    elif node.type in ("FunctionExpression", "ArrowFunctionExpression"):
        return "<anonymous>"
    return "<unknown>"


def _walk(node):
    """Recursively yield all AST nodes."""
    if not isinstance(node, esprima.nodes.Node):
        return
    yield node
    for key in node.__dict__:
        child = getattr(node, key)
        if isinstance(child, esprima.nodes.Node):
            yield from _walk(child)
        elif isinstance(child, list):
            for item in child:
                yield from _walk(item)


def extract_functions_from_js(source_code, file_path="unknown"):
    """
    Parse JavaScript source and extract functions, classes, and module-level vars.
    Returns list of chunk dicts compatible with the Python parser output.
    """
    try:
        tree = esprima.parseScript(
            source_code,
            tolerant=True,
            range=False,
            loc=True,
            comment=False
        )
    except Exception as e:
        # Try parseModule for ES module syntax (import/export)
        try:
            tree = esprima.parseModule(
                source_code,
                tolerant=True,
                loc=True,
            )
        except Exception as e2:
            print(f"Skipping {file_path}: parse error — {e2}")
            return []

    source_lines = source_code.splitlines()
    chunks = []

    def make_chunk(name, class_name, start, end, code, chunk_type, docstring=""):
        return {
            "function_name": name,
            "class_name": class_name,
            "file_path": file_path,
            "start_line": start,
            "end_line": end,
            "docstring": docstring,
            "code": code,
            "chunk_type": chunk_type
        }

    def process_body(body, class_name=None):
        for node in body:
            start = node.loc.start.line
            end = node.loc.end.line
            code = _get_source_segment(source_lines, start, end)

            # function foo() {}
            if node.type == "FunctionDeclaration":
                name = node.id.name if node.id else "<anonymous>"
                chunks.append(make_chunk(name, class_name, start, end, code, "function"))

            # const foo = () => {} or const foo = function() {}
            elif node.type == "VariableDeclaration":
                for decl in node.declarations:
                    if decl.init and decl.init.type in ("FunctionExpression", "ArrowFunctionExpression"):
                        name = decl.id.name if decl.id else "<anonymous>"
                        decl_start = decl.loc.start.line
                        decl_end = decl.loc.end.line
                        decl_code = _get_source_segment(source_lines, decl_start, decl_end)
                        chunks.append(make_chunk(name, class_name, decl_start, decl_end, decl_code, "function"))
                    else:
                        # Module-level variable
                        if class_name is None:
                            names = []
                            for decl in node.declarations:
                                if hasattr(decl.id, 'name'):
                                    names.append(decl.id.name)
                            if names:
                                chunks.append(make_chunk(
                                    f"[module] {', '.join(names)}",
                                    None, start, end, code, "module_variable"
                                ))
                        break

            # class Foo {}
            elif node.type == "ClassDeclaration":
                class_nm = node.id.name if node.id else "<anonymous>"
                class_code = "\n".join(source_lines[start - 1:min(start + 5, end)])
                chunks.append(make_chunk(f"[class] {class_nm}", None, start, end, class_code, "class_definition"))

                # Extract methods
                if node.body and node.body.body:
                    for method in node.body.body:
                        if method.type == "MethodDefinition":
                            m_start = method.loc.start.line
                            m_end = method.loc.end.line
                            m_code = _get_source_segment(source_lines, m_start, m_end)
                            m_name = method.key.name if hasattr(method.key, 'name') else "<unknown>"
                            chunks.append(make_chunk(m_name, class_nm, m_start, m_end, m_code, "function"))

            # export default function / export const foo = ...
            elif node.type == "ExportDefaultDeclaration":
                decl = node.declaration
                if decl and decl.type == "FunctionDeclaration":
                    name = decl.id.name if decl.id else "default"
                    chunks.append(make_chunk(name, class_name, start, end, code, "function"))

            elif node.type == "ExportNamedDeclaration" and node.declaration:
                decl = node.declaration
                if decl.type == "FunctionDeclaration":
                    name = decl.id.name if decl.id else "<anonymous>"
                    chunks.append(make_chunk(name, class_name, start, end, code, "function"))
                elif decl.type == "VariableDeclaration":
                    for vd in decl.declarations:
                        if vd.init and vd.init.type in ("FunctionExpression", "ArrowFunctionExpression"):
                            name = vd.id.name if vd.id else "<anonymous>"
                            chunks.append(make_chunk(name, class_name, start, end, code, "function"))

    process_body(tree.body)
    return chunks


def extract_js_from_zip(zip_file):
    """Walk a ZipFile and parse all .js files."""
    all_chunks = []

    for zip_path in zip_file.namelist():
        if not zip_path.endswith(".js"):
            continue

        with zip_file.open(zip_path) as f:
            try:
                source_code = f.read().decode("utf-8")
            except UnicodeDecodeError:
                print(f"Skipping {zip_path}: could not decode as UTF-8")
                continue

        chunks = extract_functions_from_js(source_code, file_path=zip_path)
        all_chunks.extend(chunks)

    functions = sum(1 for c in all_chunks if c["chunk_type"] == "function")
    variables = sum(1 for c in all_chunks if c["chunk_type"] == "module_variable")
    classes = sum(1 for c in all_chunks if c["chunk_type"] == "class_definition")
    print(f"JS extracted: {functions} functions, {variables} module variables, {classes} classes")

    return all_chunks