import argparse
import pathlib
import re
import sys
import textwrap

from treesitter_utils import *
# Tag Notes: di = int, df = float, du = unsigned

# Interface for translation:
# - Start with a tree-sitter function definition node
# - Translate function call expressions recursively
fn_call_translations = {}
fn_decl_translations = {}
fn_comments = {}
type_translations = {}
type_precisions = {}
constant_translations = {}
constant_types = {}
constant_comments = {}
macro_conditionals = {}
macro_conditional_translations = {}

source_file = {}
BUILTIN_TAG_NAMES = [b"df", b"di32", b"di", b"du32", b"du"]

def main():
    parser = argparse.ArgumentParser(
        description="Translate sleef source files into highway code"
    )
    parser.add_argument('sleef_src', help="Path of sleef 'src' folder")
    parser.add_argument('rename_data', help="Path of rename_data folder")
    parser.add_argument('output', help="Path to write generated header")
    
    args = parser.parse_args()

    sleef_src = pathlib.Path(args.sleef_src)
    rename_data = pathlib.Path(args.rename_data)
    out = open(args.output, 'w')


    target_functions = [
        # Single-precision ops
        "xexpf",
        "xexp2f",
        "xexp10f",
        "xexpm1f",
        "xlogf_u1",
        "xlogf",
        "xlog10f",
        "xlog2f",
        "xlog1pf",
        "xsqrtf_u05",
        "xsqrtf_u35",
        "xcbrtf",
        "xcbrtf_u1",
        "xhypotf_u05",
        "xhypotf_u35",
        "xpowf",
        "xsinf_u1",
        "xcosf_u1",
        "xtanf_u1",
        "xsinf",
        "xcosf",
        "xtanf",
        "XSINCOSF_U1",
        "XSINCOSF",
        "XSINCOSPIF_U05",
        "XSINCOSPIF_U35",
        "xsinpif_u05",
        "xcospif_u05",
        "xacosf_u1",
        "xasinf_u1",
        "xasinf",
        "xacosf",
        "xatanf",
        "xatanf_u1",
        "xatan2f_u1",
        "xatan2f",
        "xsinhf",
        "xcoshf",
        "xtanhf",
        "xsinhf_u35",
        "xcoshf_u35",
        "xtanhf_u35",
        "xasinhf",
        "xacoshf",
        "xatanhf",
        "xerff_u1",
        "xerfcf_u15",
        "xtgammaf_u1",
        "xlgammaf_u1",
        "xfmodf",
        "xremainderf",
        "xldexpf",
        "xfrfrexpf",
        "xexpfrexpf",
        "xilogbf",
        "XMODFF",
        "xnextafterf",
        "xfastsinf_u3500",
        "xfastcosf_u3500",
        "xfastpowf_u3500",
        # Double-precision ops
        "xexp",
        "xexp2",
        "xexp10",
        "xexpm1",
        "xlog_u1",
        "xlog",
        "xlog10",
        "xlog2",
        "xlog1p",
        "xsqrt_u05",
        "xsqrt_u35",
        "xcbrt",
        "xcbrt_u1",
        "xhypot_u05",
        "xhypot_u35",
        "xpow",
        "xsin_u1",
        "xcos_u1",
        "xtan_u1",
        "xsin",
        "xcos",
        "xtan",
        "xacos_u1",
        "xasin_u1",
        "xatan_u1",
        "xacos",
        "xasin",
        "xatan",
        "xatan2_u1",
        "xatan2",
        "XSINCOS_U1",
        "XSINCOS",
        "XSINCOSPI_U05",
        "XSINCOSPI_U35",
        "xsinpi_u05",
        "xcospi_u05",
        "xsinh",
        "xcosh",
        "xtanh",
        "xsinh_u35",
        "xcosh_u35",
        "xtanh_u35",
        "xasinh",
        "xacosh",
        "xatanh",
        "xerf_u1",
        "xerfc_u15",
        "xtgamma_u1",
        "xlgamma_u1",
        "xfmod",
        "xremainder",
        "xldexp",
        "xfrfrexp",
        "xexpfrexp",
        "xilogb",
        "XMODF",
        "xnextafter",
    ]

    # Read data files to register translations for simd ops, intermediate functions, and types
    for old_name, new_name, comment in parse_tsv(rename_data / "function_renames.tsv", 3):
        key, translate_fn = function_rename_translator(old_name, new_name, old_name in target_functions)
        fn_call_translations[key] = translate_fn
        fn_decl_translations[key] = translate_fn
        fn_comments[key.decode()] = comment

    for in_spec, out_spec in parse_tsv(rename_data / "simd_ops.tsv", 2):
        key, translate_fn = simd_op_translator(in_spec, out_spec)
        fn_call_translations[key] = translate_fn
    
    for in_type, out_type, precision in parse_tsv(rename_data / "types.tsv", 3):
        key, translate_fn = type_rename_translator(in_type, out_type)
        type_translations[key] = translate_fn
        type_precisions[key] = precision

    for old_name, new_name, type, comment in parse_tsv(rename_data / "constant_renames.tsv", 4):
        key, translate_fn = constant_rename_translator(old_name, new_name)
        constant_translations[key] = translate_fn
        constant_types[key] = type.encode()
        constant_comments[key] = comment.encode()

    for in_condition, out_condition in parse_tsv(rename_data / "macro_conditionals.tsv", 2):
        macro_conditionals[in_condition] = out_condition
        key, translate_fn = macro_conditonal_translator(in_condition, out_condition)
        macro_conditional_translations[key] = translate_fn

    sources = [
        "libm/sleefsimdsp.c", 
        "libm/sleefsimddp.c", 
        "common/df.h", 
        "common/dd.h",
        "common/commonfuncs.h",
        "arch/helperneon32.h",
    ]
    calls = {}
    trees = {}

    function_nodes = collections.defaultdict(list) # name: [(node, exclude_bool)]
    for s in sources:
        text = open(sleef_src / s, "rb").read()
        
        tree = c_parse(text)
        trees[s] = tree
        
        calls = {**construct_callgraph(tree.root_node), **calls}
        for f in all_defined_functions(tree.root_node):
            source_file[f] = s

        for (n, _) in c_query("(function_definition) @fn_def").captures(tree.root_node):
            exclude = False

            # Check if n is under the wrong side of a conditional definition
            if n.parent.type in ["preproc_if", "preproc_ifdef", "preproc_elif"]:
                condition = n.parent.named_children[0].text.decode()
                if macro_conditionals.get(condition) == "0":
                    exclude = True
            if n.parent.type == "preproc_else":
                condition = n.parent.parent.named_children[0].text.decode()
                if macro_conditionals.get(condition) == "1":
                    exclude = True

            name = n.child_by_field_name("declarator").child_by_field_name("declarator").text.decode()
            function_nodes[name].append((n, exclude))

    fns_to_translate = []
    for t in target_functions:
        for f in topo_sort(t, calls):
            if (f in source_file and 
                f in fn_comments
                and f not in fns_to_translate
                and f not in target_functions
                ):
                fns_to_translate.append(f)
    fns_to_translate += target_functions

    helper_code = []
    code = []
    decls = []
    for f in fns_to_translate:
        valid_nodes = [n for n, exclude in function_nodes[f] if not exclude]
        if len(valid_nodes) == 0:
            if len(function_nodes[f]) > 1:
                print(f"WARNING: found 0 valid definitions and {len(function_nodes[f])} invalid definitions for function {f} (using first invalid)", file=sys.stderr)
            node = function_nodes[f][0][0]
        else:
            node = valid_nodes[0]
        
        # Special-case for two function definitions withn a known #if ... #else ... #endif structure,
        # possibly with an intervening #elif that doesn't get used
        if len(valid_nodes) == 2 and \
            (valid_nodes[0].parent == valid_nodes[1].parent.parent or 
             valid_nodes[0].parent == valid_nodes[1].parent.parent.parent) and \
            valid_nodes[0].parent.named_children[0].text.decode() in macro_conditionals:
            translation_true = translate_sleef_function(f, valid_nodes[0])
            translation_false = translate_sleef_function(f, valid_nodes[1])
            body_true = translation_true[translation_true.find("{")+2:translation_true.rfind("}")]
            body_false = translation_false[translation_false.find("{")+2:translation_false.rfind("}")]
            translated_condition = macro_conditionals[valid_nodes[0].parent.named_children[0].text.decode()]
            maybe_newline = "" if body_false[-1] == "\n" else "\n"
            translation = translation_true.replace(
                body_true, 
                f"#if {translated_condition}\n{body_true}#else\n{body_false}{maybe_newline}#endif\n"
            )
        else:
            if len(valid_nodes) > 1:
                print(f"WARNING: found {len(valid_nodes)} valid definitions for function {f} (using first valid)", file=sys.stderr)
            translation = translate_sleef_function(f, node)


        if f in target_functions:
            code.append(translation)
            decls.append(translation[:translation.find("{")].strip() + ";")
        else:
            helper_code.append(translation)
    
    sleef_constant_defs = open(sleef_src / "common/misc.h", "rb").read()
    const_defs = translate_constant_defs(c_parse(sleef_constant_defs).root_node).decode()

    print(FILE_TEMPLATE.format(
        decls="\n\n".join(decls),
        const_defs=const_defs,
        helper_code="\n\n".join(helper_code),
        code="\n\n".join(code),
    ), file=out)

def translate_sleef_function(function_name, node):
    output_template = textwrap.dedent("""
    // {comment}
    // Translated from {file}:{line} {old_function_name}
    template<class D>
    HWY_INLINE {translated_function}
    """).strip()

    file = source_file[function_name]
    line = node.start_point[0] + 1
    [(decl, _)] = c_query("(function_definition declarator: (function_declarator) @decl)").captures(node)
    # Manually strip off some of the macros in the function definition
    start_pos = decl.start_byte - node.start_byte
    start_pos = node.text.rfind(b" ", 0, start_pos)
    start_pos = node.text.rfind(b" ", 0, start_pos)
    node = c_parse(node.text[start_pos+1:]).root_node.children[0]
    return output_template.format(
        comment = fn_comments[function_name],
        file = file,
        line = line,
        old_function_name = function_name,
        translated_function = translate_function(node).decode()
    )

def parse_tsv(path, count):
    for l in open(path):
        l = l.strip()
        if "#" in l:
            l = l[:l.find("#")]
        if len(l) == 0:
            continue
        res = l.split("\t")
        if len(res) != count:
            print(f"Error: got \"{res}\" with {len(res)} fields instead of {count}", file=sys.stderr)
        yield res

def simd_op_translator(in_spec, out_spec):
    in_spec = c_parse(in_spec.encode() + b";").root_node.children[0].children[0]
    out_spec = cpp_parse(out_spec.encode() + b";").root_node.children[0].children[0]
        
    c_args = c_query("(call_expression arguments: (argument_list (identifier) @arg))")
    in_args = [n.text for (n, _) in c_args.captures(in_spec)]

    c_fn_name = c_query("(call_expression function: (identifier) @fn_name)")
    (in_fn_name,) = [n.text for (n, _) in c_fn_name.captures(in_spec)]

    cpp_args = cpp_query("""
    (type_identifier) @arg
    (identifier) @arg
    """)

    tag_types = set()
    summary = []
    last_offset = 0
    for n, tag in cpp_args.captures(out_spec):
        summary.append(out_spec.text[last_offset:n.start_byte])
        if tag == "arg":
            if n.text in in_args:
                summary.append(in_args.index(n.text))
            else:
                summary.append(n.text)
                if n.text in BUILTIN_TAG_NAMES:
                    tag_types.add(n.text)
        last_offset = n.end_byte
    summary.append(out_spec.text[last_offset:])

    def translate(node, ctx):
        assert node.type == "call_expression" 
        assert node.child_by_field_name("function").text == in_fn_name
        arg_nodes = node.child_by_field_name("arguments").named_children
        ctx["tag_types"] |= tag_types
        return b"".join(
            translate_tree(arg_nodes[i], ctx) if type(i) is int else i
            for i in summary
        )
    return (in_fn_name, translate)

def function_rename_translator(old_name, new_name, is_top_level=False):
    old_name = old_name.encode()
    new_name = new_name.encode()
    def translate(node, ctx):
        assert node.type == "call_expression" or node.type == "function_declarator"
        if node.type == "call_expression":
            assert node.child_by_field_name("function").text == old_name
            # Pass tag as first parameter, and mark that we need the tag
            ctx["tag_types"].add(b"df")
            ret = new_name + b"(df, " + translate_tree(node.child_by_field_name("arguments"), ctx).lstrip(b"(")
            if is_top_level:
                return b"sleef::" + ret
            else:
                return ret
        if node.type == "function_declarator":
            assert node.child_by_field_name("declarator").text == old_name
            # Add in tag as first parameter
            return new_name + b"(const D df, " + translate_tree(node.child_by_field_name("parameters"), ctx).lstrip(b"(")

    return (old_name, translate)

def translate_fn_call(node, ctx):
    assert node.type == "call_expression" 
    fn_name = node.child_by_field_name("function").text
    if fn_name in fn_call_translations:
        return fn_call_translations[fn_name](node, ctx)
    else:
        return None

def translate_fn_decl(node, ctx):
    assert node.type == "function_declarator"
    fn_name = node.child_by_field_name("declarator").text
    if fn_name in fn_decl_translations:
        return fn_decl_translations[fn_name](node, ctx)
    else:
        return None

def type_rename_translator(old_name, new_name):
    old_name = old_name.encode()
    new_name = new_name.encode()
    def translate(node, ctx):
        assert node.type == "type_identifier"
        assert node.text == old_name
        return new_name
    return (old_name, translate)

def translate_type_id(node, ctx):
    assert node.type == "type_identifier"
    if node.text in type_translations:
        return type_translations[node.text](node, ctx)
    else:
        return None
    
def constant_rename_translator(old_name, new_name):
    old_name = old_name.encode()
    new_name = new_name.encode()
    def translate(node, ctx):
        assert node.type == "identifier"
        assert node.text == old_name
        return new_name
    return (old_name, translate)

def translate_constant(node, ctx):
    assert node.type == "identifier"
    if node.text in constant_translations:
        return constant_translations[node.text](node, ctx)
    else:
        return None

def macro_conditonal_translator(in_condition, out_condition):
    in_condition = in_condition.encode()
    out_condition = out_condition.encode()
    def translate(node, ctx):
        assert node.type in ["preproc_if", "preproc_ifdef"]
        # Verify assumptions about the early nodes in the children
        if node.type == "preproc_if":
            assert node.children[0].text == b"#if"    
            assert node.field_name_for_child(1) == "condition"
            assert node.children[2].text == b"\n"
            first_line_nodes = 3
        else:
            assert node.children[0].text in [b"#ifdef", b"#ifndef"]
            assert node.field_name_for_child(1) == "name"
            assert node.children[2].is_named
            first_line_nodes = 2
            if node.children[0].text == b"#ifndef":
                # We could presumably swap 0 and 1 for ifndef, but skip for now
                # since it hasn't come up yet in practice
                assert out_condition not in [b"0", b"1"]
        

        if out_condition == b"0":
            alt = node.child_by_field_name("alternative")
            if alt is None:
                return b""
            alt = translate_tree(alt, ctx)
            return alt[alt.find(b"\n")+1:]
        
        last_byte = node.start_byte
        child_text = []
        for c in node.children:
            child_text.append(ctx["text"][last_byte:c.start_byte])
            last_byte=c.end_byte
            child_text.append(translate_tree(c, ctx))
        child_text.append(ctx["text"][last_byte:node.end_byte])
        
        if out_condition == b"1":
            [alt_index] = [i for i, _  in enumerate(node.children) if node.field_name_for_child(i) == "alternative"]
            # child_text is [literal, child_0, literal, child_1, ... literal, child_n, literal]
            child_text = child_text[(1 + 2*first_line_nodes):(1 + 2*alt_index)]
            return b"".join(child_text)
        else:
            # Condition text should be position 3
            child_text[3] = out_condition
        return b"".join(child_text)

    return (in_condition, translate)

def translate_macro_conditional(node, ctx):
    assert node.type in ["preproc_if", "preproc_ifdef"]
    condition = node.named_children[0].text
    if condition in macro_conditional_translations:
        return macro_conditional_translations[condition](node, ctx)
    else:
        return None

def translate_tag_name_conflict(node, ctx):
    assert node.type == "identifier"
    if node.text in BUILTIN_TAG_NAMES:
        return node.text + b"_"
    return node.text

def translate_pointer_argument(node, ctx):
    assert node.type == "pointer_declarator"
    return node.text.replace(b"*", b"&")

def ancestors(node):
    while node.parent is not None:
        node = node.parent
        yield node

def translate_function(node):
    assert node.type == "function_definition"

    root_node = list(ancestors(node))[-1]

    handlers = {
        "fn_call": translate_fn_call,
        "type_id": translate_type_id,
        "fn_decl": translate_fn_decl,
        "const_id": translate_constant,
        "macro_conditional": translate_macro_conditional,
        "tag_name_conflict": translate_tag_name_conflict,
        "pointer_argument": translate_pointer_argument,
    }

    captures = c_query(
    """
    (call_expression) @fn_call
    (type_identifier) @type_id
    (function_declarator) @fn_decl
    (identifier) @const_id
    (preproc_if) @macro_conditional
    (preproc_ifdef) @macro_conditional
    ((identifier) @tag_name_conflict
        (#match? @tag_name_conflict "^d[iu](32)?$"))
    (parameter_declaration (pointer_declarator) @pointer_argument)
    """
    ).captures(node)
    fn_body = node.child_by_field_name("body")
    
    declared_identifiers = set(
        n.text for n, _ in c_query("(_ declarator: (identifier) @id)").captures(node)
    )
    declared_identifiers |= set(fn_call_translations.keys())
    
    unknown_ids = set()
    for n, _ in c_query("(identifier) @id").captures(fn_body):
        if n.text in declared_identifiers:
            continue
        if n.text in fn_call_translations:
            continue
        if n.text in constant_translations:
            continue
        if n.text.decode() in macro_conditionals:
            continue
        
        is_known_macro = False
        prev_a = n
        for a in ancestors(n):
            if a.type in ["preproc_if", "preproc_ifdef"]:
                is_known_macro = a.named_children[0].text in macro_conditional_translations and prev_a == a.named_children[0]
                break    
            prev_a = a
        if is_known_macro:
            continue

        unknown_ids.add(n.text)

    
    if len(unknown_ids) > 0:
        name = node.child_by_field_name("declarator").child_by_field_name("declarator").text.decode()
        print(f"WARNING: Possibly unknown identifiers in {name}: ", b", ".join(unknown_ids).decode(), file=sys.stderr)

    ctx = make_translation_context(root_node, handlers, captures)
    ctx["tag_types"] = set()

    # Determine if function should be restricted to floats or doubles:
    # 1. If return type is float/double, use that
    # 2. If return type is ambiguous, use type of first float/double argument
    return_type = translate_tree(node.child_by_field_name("type"), ctx).decode()
    type_nodes = [node.child_by_field_name("type")] + \
        [n for n, _ in c_query("(parameter_declaration (type_identifier) @type)").captures(node.child_by_field_name("declarator"))]
    for n in type_nodes:
        if "float" == type_precisions[n.text]:
            return_type = f"HWY_SLEEF_IF_FLOAT(D, {return_type})".encode()
            break
        elif "double" == type_precisions[n.text]:
            return_type = f"HWY_SLEEF_IF_DOUBLE(D, {return_type})".encode()
            break
    if not return_type.startswith(b"HWY_SLEEF_IF"):
        print(f"WARNING: Could not determine return type precision for function: {node.child_by_field_name('declarator').text.decode()}")

    signature = translate_tree(node.child_by_field_name("declarator"), ctx)
    body = translate_tree(node.child_by_field_name("body"), ctx)
    
    
    
    tag_defs = []
    for tag in ctx["tag_types"]:
        if tag == b"df":
            continue # Coming in via parameters
        elif tag == b"di":
            tag_defs.append(b"  RebindToSigned<D> di;")
        elif tag == b"di32":
            tag_defs.append(b"  RebindToSigned32<D> di32;")
        elif tag == b"du":
            tag_defs.append(b"  RebindToUnsigned<D> du;")
        elif tag == b"du32":
            tag_defs.append(b"  RebindToUnsigned32<D> du32;")
        else:
            assert False
    tag_defs = b"\n".join(tag_defs)
    if len(tag_defs) > 0:
        tag_defs += b"\n  \n"
    body = b"{\n" + tag_defs + body.lstrip(b"{\n")
    
    return postprocess_function(b" ".join([return_type, signature, body]))

def make_translation_context(node, handlers, captures):
    # We make a list of handlers to handle cases where >1 capture is relevant,
    # which so far happens for number literals in post-processing.
    # `translate_tree` will apply the first handler which returns a result.
    capture_handler = {}
    for n, t in captures:
        if n.id not in capture_handler:
            capture_handler[n.id] = [handlers[t]]
        else:
            capture_handler[n.id].append(handlers[t])
    
    return {
        "text": node.text,
        "capture_handler": capture_handler,
        "has_child_match": set(n.id for c in captures for n in ancestors(c[0])),
    }

def translate_tree(node, ctx):
    # If the current node is a match, call the appropriate handler
    if node.id in ctx["capture_handler"]:
        for f in ctx["capture_handler"][node.id]:
            res = f(node, ctx)
            if res is not None:
                return res

    if node.id not in ctx["has_child_match"]:
        return node.text

    # Recurse into children, preserving all text outside of the child nodes themselves
    last_byte = node.start_byte
    child_text = []
    for c in node.children:
        child_text.append(ctx["text"][last_byte:c.start_byte])
        last_byte=c.end_byte
        child_text.append(translate_tree(c, ctx))
    child_text.append(ctx["text"][last_byte:node.end_byte])
    return b"".join(child_text)


def translate_preproc_define(node, ctx):
    assert node.type == "preproc_def"
    name = node.child_by_field_name("name")
    value = node.child_by_field_name("value")
    if name.text in constant_translations:
        # Translate the value portion separately since tree-sitter doesn't parse it as code
        value_node = c_parse(value.text).root_node
        handlers = {
            "const_id": translate_constant,
        }
        captures = c_query("(identifier) @const_id").captures(value_node)
        local_ctx = make_translation_context(value_node, handlers, captures)
        value = translate_tree(value_node, local_ctx)
        if b"//" in value:
            value = value[:value.find(b"//")]
        return (
            b" ".join([b"constexpr", constant_types[name.text], translate_tree(name, ctx), b"=", value + b";"]) +
            b" // " + constant_comments[name.text] + b"\n" 
        )
    else:
        return b""

def translate_preproc_if(node, ctx):
    assert node.type == "preproc_if"
    children = [n for n in node.children if n.type=="preproc_def"]
    # Note: we ignore the "alternative" branch in #if #else, since if 
    # nothing is defined under the true case, we can probably ignore the false case

    # Cut out if there aren't any useful defines below it
    if all(len(translate_tree(c, ctx)) == 0 for c in children):
        return b""

    # Otherwise, defer to normal translation setup
    return None
        
           

def translate_constant_defs(root_node):
    """Process constant macro definitions and return a translated copy of the code"""
    handlers = {
        "define": translate_preproc_define,
        "const_id": translate_constant,
        "preproc_if": translate_preproc_if,
        "preproc_strip": lambda n, ctx: b"",
    }

    captures = c_query(
    """
    (preproc_def) @define
    (identifier) @const_id
    (preproc_if) @preproc_if
    (preproc_function_def) @preproc_strip
    (preproc_ifdef) @preproc_strip

    (preproc_if (comment) @preproc_strip)
    (preproc_elif (comment) @preproc_strip)
    """
    ).captures(root_node)

    ctx = make_translation_context(root_node, handlers, captures)

    query_top_level = c_query(
    """
    (translation_unit (preproc_ifdef (preproc_if) @if))
    (translation_unit (preproc_ifdef (preproc_def) @define))
    (translation_unit (preproc_ifdef (preproc_ifdef (preproc_def) @define)))
    (translation_unit (preproc_ifdef (preproc_if (preproc_ifdef (preproc_def) @define))))
    """
    )
    translations = []
    for n, tag in query_top_level.captures(root_node):
        if tag == "define":
            translations.append(translate_tree(n, ctx).strip())
        if tag == "if":
            res = translate_tree(n, ctx)
            res = re.sub(b'\n[\n ]+', b'\n', res)
            translations.append(res)
    translations = [t for t in translations if len(t) > 0]
    return b"\n".join(translations)


def topo_sort(start, callgraph):
    """Return a topologically-sorted list from a callgraph (inefficient implementation)"""
    if start not in callgraph:
        return [start]
    
    # Use sorted children to try to achieve a more stable ordering
    res = []
    for child in sorted(callgraph[start]):
        res += [x for x in topo_sort(child, callgraph) if x not in res]
    
    res.append(start)
    return res

def postprocess_function(text):
    node = cpp_parse(text).root_node

    def noop(n, ctx):
        return None
    
    handlers = {
        "shiftright_id": noop,
        "shiftright_arg": postprocess_shiftright_arg if b"HWY_SLEEF_IF_DOUBLE" in text else noop,
        "redundant_cast_outer": noop,
        "redundant_cast_inner": noop,
        "redundant_cast_call": postprocess_redundant_cast,
        "estrin_call": noop,
        "estrin_arg": postprocess_estrin_arg,
        "comment": postprocess_comment,
        "macro_variable_decl": postprocess_macro_variable_decl,
        "constant": postprocess_constant,
    }
    captures = cpp_query(
    """
    (template_function 
        name: (identifier) @shiftright_id 
        arguments: (template_argument_list (number_literal) @shiftright_arg)
        (#eq? @shiftright_arg 31)
        (#eq? @shiftright_id ShiftRight))
    (call_expression 
        function: (identifier) @redundant_cast_outer
        arguments: (argument_list (call_expression function: (identifier) @redundant_cast_inner)) @redundant_cast_call
        (#eq? @redundant_cast_outer @redundant_cast_inner)
        (#match? @redundant_cast_outer "BitCast|RebindMask"))
    (call_expression
        function: (identifier) @estrin_call
        arguments: (argument_list (call_expression) @estrin_arg)
        (#eq? @estrin_call Estrin))

    (comment) @comment
    (preproc_call) @macro_variable_decl
    (preproc_function_def) @macro_variable_decl
    (number_literal) @constant
    """
    ).captures(node)
    ctx = make_translation_context(node, handlers, captures)
    return translate_tree(node, ctx)

def postprocess_shiftright_arg(node, ctx):
    assert node.type == "number_literal"
    if node.text == b"31":
        return b"63"
    return None

def postprocess_redundant_cast(node, ctx):
    """Remove the inner function call since we're just doing two casts right after the other"""
    assert node.type == "argument_list"
    assert node.parent.child_by_field_name("function").text in [b"BitCast", b"RebindMask"]
    assert node.named_children[1].child_by_field_name("function").text in [b"BitCast", b"RebindMask"]
    second_arg = node.named_children[1]
    return node.text[:(second_arg.start_byte-node.start_byte)] + \
           node.named_children[1].child_by_field_name("arguments").named_children[1].text + b")"

def postprocess_estrin_arg(node, ctx):
    """
    This helps clean up Erf, where the simd_op entry for Estrin
    inserts Set(df, ...) calls even though the arguments are already
    vectors returned by IfThenElse.
    """
    assert node.type == "call_expression"
    if node.child_by_field_name("function").text != b"Set":
        return None
    if len(node.child_by_field_name("arguments").named_children) != 2:
        return None
    second_arg = node.child_by_field_name("arguments").named_children[1]
    if second_arg.type != "call_expression":
        return None
    if second_arg.child_by_field_name("function").text != b"IfThenElse":
        return None
    return translate_tree(second_arg, ctx)

def postprocess_comment(node, ctx):
    """Remove spurious comments that might not make sense in the translated code"""
    # No empty comments
    if len(node.text.strip()) == 2:
        return b""
    # No comments marking ends of #if blocks
    if b"#if" in node.text:
        return b""
    # No multiline comments
    if node.start_point[0] != node.end_point[0]:
        return b""
    return None

def postprocess_macro_variable_decl(node, ctx):
    if b"C2V" in node.text:
        return b""
    else:
        print(f"WARNING: macro variable op in function body: '{node.text.decode().strip()}'", file=sys.stderr)
    return None

def postprocess_constant(node, ctx):
    # This is used as a cutoff for returning infinity in Pow and Exp for 
    # double-precision, but I think the SLEEF constant might be slightly too 
    # low here
    if node.text == b"709.78271114955742909217217426":
        return b"709.78271289338399673222338991"
    return None

FILE_TEMPLATE = textwrap.dedent(
"""
// This file is translated from the SLEEF vectorized math library.
// Translation performed by Ben Parks copyright 2024.
// Translated elements available under the following licenses, at your option:
//   BSL-1.0 (http://www.boost.org/LICENSE_1_0.txt),
//   MIT (https://opensource.org/license/MIT/), and
//   Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
// 
// Original SLEEF copyright:
//   Copyright Naoki Shibata and contributors 2010 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if defined(HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_) == \\
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_
#undef HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_
#else
#define HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_
#endif

#include <type_traits>
#include "hwy/highway.h"
#include "Estrin.h"
#include "AVX512FloatUtils.h"

extern const float PayneHanekReductionTable_float[];
extern const double PayneHanekReductionTable_double[];

HWY_BEFORE_NAMESPACE();
namespace hwy {{
namespace HWY_NAMESPACE {{
namespace sleef {{

#undef HWY_SLEEF_HAS_FMA
#if (HWY_ARCH_X86 && HWY_TARGET < HWY_SSE4) || HWY_ARCH_ARM || HWY_ARCH_S390X || HWY_ARCH_RVV 
#define HWY_SLEEF_HAS_FMA 1
#endif

#undef HWY_SLEEF_IF_DOUBLE
#define HWY_SLEEF_IF_DOUBLE(D, V) typename std::enable_if<std::is_same<double, TFromD<D>>::value, V>::type
#undef HWY_SLEEF_IF_FLOAT
#define HWY_SLEEF_IF_FLOAT(D, V) typename std::enable_if<std::is_same<float, TFromD<D>>::value, V>::type

{decls}

namespace {{

//////////////////
// Constants
//////////////////
{const_defs}


{helper_code}

}}

{code}

}}  // namespace sleef
}}  // namespace HWY_NAMESPACE
}}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_
""")


if __name__ == "__main__":
    main()