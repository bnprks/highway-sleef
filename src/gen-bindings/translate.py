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
constant_translations = {}
constant_types = {}
constant_comments = {}

def main():
    parser = argparse.ArgumentParser(
        description="Translate sleef source files into highway code"
    )
    parser.add_argument('sleef_src', help="Path of sleef 'src' folder")
    parser.add_argument('rename_data', help="Path of rename_data folder")
    parser.add_argument('treesitter_lib', help="Path for compiled treesitter grammar")
    parser.add_argument('output', help="Path to write generated header")
    
    args = parser.parse_args()

    sleef_src = pathlib.Path(args.sleef_src)
    rename_data = pathlib.Path(args.rename_data)
    out = open(args.output, 'w')

    set_treesitter_lib(args.treesitter_lib)

    # Read data files to register translations for simd ops, intermediate functions, and types
    for old_name, new_name, comment in parse_tsv(rename_data / "function_renames.tsv", 3):
        key, translate_fn = function_rename_translator(old_name, new_name)
        fn_call_translations[key] = translate_fn
        fn_decl_translations[key] = translate_fn
        fn_comments[key.decode()] = comment

    for in_spec, out_spec in parse_tsv(rename_data / "simd_ops.tsv", 2):
        key, translate_fn = simd_op_translator(in_spec, out_spec)
        fn_call_translations[key] = translate_fn
    
    for in_type, out_type in parse_tsv(rename_data / "types.tsv", 2):
        key, translate_fn = type_rename_translator(in_type, out_type)
        type_translations[key] = translate_fn

    for old_name, new_name, type, comment in parse_tsv(rename_data / "constant_renames.tsv", 4):
        key, translate_fn = constant_rename_translator(old_name, new_name)
        constant_translations[key] = translate_fn
        constant_types[key] = type.encode()
        constant_comments[key] = comment.encode()

    text = b"\n".join([
        open(sleef_src / "libm/sleefsimdsp.c", "rb").read(),
        open(sleef_src / "common/df.h", "rb").read(),
    ])

    sources = ["libm/sleefsimdsp.c", "common/df.h"]
    calls = {}
    source_file = {}
    trees = {}

    function_nodes = {}
    for s in sources:
        text = open(sleef_src / s, "rb").read()
        
        tree = c_parse(text)
        trees[s] = tree
        
        calls = {**construct_callgraph(tree.root_node), **calls}
        for f in all_defined_functions(tree.root_node):
            source_file[f] = s

        for (n, _) in c_query("(function_definition) @fn_def").captures(tree.root_node):
            name = n.child_by_field_name("declarator").child_by_field_name("declarator").text.decode()
            function_nodes[name] = n

    target_functions = ["xexpf", "xexpm1f"]
    # target_functions = ["xpowf"]
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

    output_template = textwrap.dedent("""
    // {comment}
    // Translated from {file}:{line} {old_function_name}
    template<class D>
    HWY_INLINE {translated_function}
    """).strip()

    helper_code = []
    code = []
    decls = []
    for f in fns_to_translate:
        file = source_file[f]

        # Manually strip off some of the macros in the function definition
        node = function_nodes[f]
        line = node.start_point[0] + 1
        [(decl, _)] = c_query("(function_definition declarator: (function_declarator) @decl)").captures(node)
        start_pos = decl.start_byte - node.start_byte
        start_pos = node.text.rfind(b" ", 0, start_pos)
        start_pos = node.text.rfind(b" ", 0, start_pos)
        node = c_parse(node.text[start_pos+1:]).root_node.children[0]
        translation = output_template.format(
            comment = fn_comments[f],
            file = file,
            line = line,
            old_function_name = f,
            translated_function = translate_function(node).decode()
        )
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
    (template_function 
        arguments: (template_argument_list (type_descriptor (type_identifier) @arg)))
    (call_expression arguments: (argument_list (identifier) @arg))
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
                if len(n.text) == 2 and n.text[0] == ord('d'):
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

def function_rename_translator(old_name, new_name):
    old_name = old_name.encode()
    new_name = new_name.encode()
    def translate(node, ctx):
        assert node.type == "call_expression" or node.type == "function_declarator"
        if node.type == "call_expression":
            assert node.child_by_field_name("function").text == old_name
            # Pass tag as first parameter, and mark that we need the tag
            ctx["tag_types"].add(b"df")
            return new_name + b"(df, " + translate_tree(node.child_by_field_name("arguments"), ctx).lstrip(b"(")
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
    }

    captures = c_query(
    """
    (call_expression) @fn_call
    (type_identifier) @type_id
    (function_declarator) @fn_decl
    (argument_list (identifier) @const_id)
    (argument_list (unary_expression (identifier) @const_id))
    """
    ).captures(node)
    fn_body = node.child_by_field_name("body")
    
    declared_identifiers = set(
        n.text for n, _ in c_query("(_ declarator: (identifier) @id)").captures(node)
    )
    declared_identifiers |= set(fn_call_translations.keys())
    
    unknown_ids = []
    for n, _ in c_query("(identifier) @id").captures(fn_body):
        if n.text in declared_identifiers:
            continue
        if n.text in fn_call_translations:
            continue
        if n.text in constant_translations:
            if n.parent.type == "argument_list":
                continue
            if n.parent.type == "unary_expression" and n.parent.parent.type == "argument_list":
                continue
        unknown_ids.append(n.text)

    
    if len(unknown_ids) > 0:
        print("WARNING: Possibly unknown identifiers: ", b", ".join(unknown_ids), file=sys.stderr)

    ctx = {
        "text": root_node.text,
        "capture_handler": {n.id: handlers[t] for (n, t) in captures},
        "has_child_match": set(n.id for c in captures for n in ancestors(c[0])),
        "tag_types": set(),
    }

    return_type = translate_tree(node.child_by_field_name("type"), ctx)
    signature = translate_tree(node.child_by_field_name("declarator"), ctx)
    body = translate_tree(node.child_by_field_name("body"), ctx)
    
    tag_defs = []
    for tag in ctx["tag_types"]:
        if tag == b"df":
            continue # Coming in via parameters
        elif tag == b"di":
            tag_defs.append(b"  RebindToSigned<D> di;")
        elif tag == b"du":
            tag_defs.append(b"  RebindToUnsigned<D> du;")
        else:
            assert False
    tag_defs = b"\n".join(tag_defs)
    if len(tag_defs) > 0:
        tag_defs += b"\n  \n"
    body = b"{\n" + tag_defs + body.lstrip(b"{\n")
    
    return b" ".join([return_type, signature, body])

def translate_tree(node, ctx):
    # If the current node is a match, call the appropriate handler
    if node.id in ctx["capture_handler"]:
        res = ctx["capture_handler"][node.id](node, ctx)
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
    # breakpoint()
    if name.text in constant_translations:
        return (
            b" ".join([b"constexpr", constant_types[name.text], translate_tree(name, ctx), b"=", translate_tree(value, ctx) + b";"]) +
            b" // " + constant_comments[name.text] + b"\n" 
        )
    else:
        return b""

def translate_preproc_if(node, ctx):
    assert node.type == "preproc_if"
    children = [n for n in node.children if n.type=="preproc_def"]
    # breakpoint()

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

    ctx = {
        "text": root_node.text,
        "capture_handler": {n.id: handlers[t] for (n, t) in captures},
        "has_child_match": set(n.id for c in captures for n in ancestors(c[0])),
        
    }

    query_top_level = c_query(
    """
    (translation_unit (preproc_ifdef (preproc_if) @if))
    (translation_unit (preproc_ifdef (preproc_def) @define))
    """
    )
    translations = []
    for n, tag in query_top_level.captures(root_node):
        if tag == "define":
            translations.append(translate_tree(n, ctx).strip())
        if tag == "if":
            # breakpoint()
            res = translate_tree(n, ctx)
            res = re.sub(b'\n[\n ]+', b'\n', res)
            translations.append(res)
    translations = [t for t in translations if len(t) > 0]
    return b"\n".join(translations)


def topo_sort(start, callgraph):
    """Return a topologically-sorted list from a callgraph (inefficient implementation)"""
    if start not in callgraph:
        return [start]
    
    res = []
    for child in callgraph[start]:
        res += [x for x in topo_sort(child, callgraph) if x not in res]
    
    res.append(start)
    return res

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

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {{
namespace HWY_NAMESPACE {{
namespace sleef {{

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

#endif  // HIGHWAY_HWY_CONTRIB_MATH_MATH_INL_H_
""")


if __name__ == "__main__":
    main()