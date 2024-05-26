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
        "xsinhf",
        "xcoshf",
        "xtanhf",
        "xsinhf_u35",
        "xcoshf_u35",
        "xtanhf_u35",
        "xacosf_u1",
        "xasinf_u1",
        "xasinhf",
        "xacosf",
        "xasinf",
        "xatanf",
        "xacoshf",
        "xatanf_u1",
        "xatanhf",
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
        "xsinh",
        "xcosh",
        "xtanh"
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
        if f == "xsqrtf_u35":
            pass #breakpoint()
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
            assert node.children[0].text == b"#ifdef"
            assert node.field_name_for_child(1) == "name"
            assert node.children[2].is_named
            first_line_nodes = 2
        

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

    ctx = {
        "text": root_node.text,
        "capture_handler": {n.id: handlers[t] for (n, t) in captures},
        "has_child_match": set(n.id for c in captures for n in ancestors(c[0])),
        "tag_types": set(),
    }

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
        value = translate_tree(value, ctx)
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

    ctx = {
        "text": root_node.text,
        "capture_handler": {n.id: handlers[t] for (n, t) in captures},
        "has_child_match": set(n.id for c in captures for n in ancestors(c[0])),
        
    }

    query_top_level = c_query(
    """
    (translation_unit (preproc_ifdef (preproc_if) @if))
    (translation_unit (preproc_ifdef (preproc_def) @define))
    (translation_unit (preproc_ifdef (preproc_ifdef (preproc_def) @define)))
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

#include <type_traits>
#include "hwy/highway.h"

extern const float PayneHanekReductionTable_float[]; // Precomputed table of exponent values for Payne Hanek reduction
extern const double PayneHanekReductionTable_double[]; // Precomputed table of exponent values for Payne Hanek reduction

HWY_BEFORE_NAMESPACE();
namespace hwy {{
namespace HWY_NAMESPACE {{

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
HWY_API Vec512<float> GetExponent(Vec512<float> x) {{
  return Vec512<float>{{_mm512_getexp_ps(x.raw)}};
}}
HWY_API Vec256<float> GetExponent(Vec256<float> x) {{
  return Vec256<float>{{_mm256_getexp_ps(x.raw)}};
}}
template<size_t N>
HWY_API Vec128<float, N> GetExponent(Vec128<float, N> x) {{
  return Vec128<float, N>{{_mm_getexp_ps(x.raw)}};
}}

HWY_API Vec512<double> GetExponent(Vec512<double> x) {{
  return Vec512<double>{{_mm512_getexp_pd(x.raw)}};
}}
HWY_API Vec256<double> GetExponent(Vec256<double> x) {{
  return Vec256<double>{{_mm256_getexp_pd(x.raw)}};
}}
template<size_t N>
HWY_API Vec128<double, N> GetExponent(Vec128<double, N> x) {{
  return Vec128<double, N>{{_mm_getexp_pd(x.raw)}};
}}

HWY_API Vec512<float> GetMantissa(Vec512<float> x) {{
  return Vec512<float>{{_mm512_getmant_ps(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)}};
}}
HWY_API Vec256<float> GetMantissa(Vec256<float> x) {{
  return Vec256<float>{{_mm256_getmant_ps(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)}};
}}
template<size_t N>
HWY_API Vec128<float, N> GetMantissa(Vec128<float, N> x) {{
  return Vec128<float, N>{{_mm_getmant_ps(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)}};
}}

HWY_API Vec512<double> GetMantissa(Vec512<double> x) {{
  return Vec512<double>{{_mm512_getmant_pd(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)}};
}}
HWY_API Vec256<double> GetMantissa(Vec256<double> x) {{
  return Vec256<double>{{_mm256_getmant_pd(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)}};
}}
template<size_t N>
HWY_API Vec128<double, N> GetMantissa(Vec128<double, N> x) {{
  return Vec128<double, N>{{_mm_getmant_pd(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)}};
}}

template<int I>
HWY_API Vec512<float> Fixup(Vec512<float> a, Vec512<float> b, Vec512<int> c) {{
    return Vec512<float>{{_mm512_fixupimm_ps(a.raw, b.raw, c.raw, I)}};
}}
template<int I>
HWY_API Vec256<float> Fixup(Vec256<float> a, Vec256<float> b, Vec256<int> c) {{
    return Vec256<float>{{_mm256_fixupimm_ps(a.raw, b.raw, c.raw, I)}};
}}
template<int I, size_t N>
HWY_API Vec128<float, N> Fixup(Vec128<float, N> a, Vec128<float, N> b, Vec128<int, N> c) {{
    return Vec128<float, N>{{_mm_fixupimm_ps(a.raw, b.raw, c.raw, I)}};
}}

template<int I>
HWY_API Vec512<double> Fixup(Vec512<double> a, Vec512<double> b, Vec512<int64_t> c) {{
    return Vec512<double>{{_mm512_fixupimm_pd(a.raw, b.raw, c.raw, I)}};
}}
template<int I>
HWY_API Vec256<double> Fixup(Vec256<double> a, Vec256<double> b, Vec256<int64_t> c) {{
    return Vec256<double>{{_mm256_fixupimm_pd(a.raw, b.raw, c.raw, I)}};
}}
template<int I, size_t N>
HWY_API Vec128<double, N> Fixup(Vec128<double, N> a, Vec128<double, N> b, Vec128<int64_t, N> c) {{
    return Vec128<double, N>{{_mm_fixupimm_pd(a.raw, b.raw, c.raw, I)}};
}}
#endif

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

template<class D>
using RebindToSigned32 = Rebind<int32_t, D>;
template<class D>
using RebindToUnsigned32 = Rebind<uint32_t, D>;

// Estrin's Scheme is a faster method for evaluating large polynomials on
// super scalar architectures. It works by factoring the Horner's Method
// polynomial into power of two sub-trees that can be evaluated in parallel.
// Wikipedia Link: https://en.wikipedia.org/wiki/Estrin%27s_scheme
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1) {{
  return MulAdd(c1, x, c0);
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T c0, T c1, T c2) {{
  return MulAdd(x2, c2, MulAdd(c1, x, c0));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T c0, T c1, T c2, T c3) {{
  return MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3, T c4) {{
  return MulAdd(x4, c4, MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3, T c4, T c5) {{
  return MulAdd(x4, MulAdd(c5, x, c4),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6) {{
  return MulAdd(x4, MulAdd(x2, c6, MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7) {{
  return MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8) {{
  return MulAdd(x8, c8,
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9) {{
  return MulAdd(x8, MulAdd(c9, x, c8),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10) {{
  return MulAdd(x8, MulAdd(x2, c10, MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11) {{
  return MulAdd(x8, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12) {{
  return MulAdd(
      x8, MulAdd(x4, c12, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
      MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
             MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13) {{
  return MulAdd(x8,
                MulAdd(x4, MulAdd(c13, x, c12),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14) {{
  return MulAdd(x8,
                MulAdd(x4, MulAdd(x2, c14, MulAdd(c13, x, c12)),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15) {{
  return MulAdd(x8,
                MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15, T c16) {{
  return MulAdd(
      x16, c16,
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15, T c16, T c17) {{
  return MulAdd(
      x16, MulAdd(c17, x, c16),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15, T c16, T c17,
                                     T c18) {{
  return MulAdd(
      x16, MulAdd(x2, c18, MulAdd(c17, x, c16)),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}}

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

#if HWY_ONCE
 __attribute__((aligned(64)))
const float PayneHanekReductionTable_float[] = {{
    // clang-format off
  0.159154892, 5.112411827e-08, 3.626141271e-15, -2.036222915e-22,
  0.03415493667, 6.420638243e-09, 7.342738037e-17, 8.135951656e-24,
  0.03415493667, 6.420638243e-09, 7.342738037e-17, 8.135951656e-24,
  0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
  0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
  0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
  0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
  0.0009518179577, 1.342109202e-10, 1.791623576e-17, 1.518506657e-24,
  0.0009518179577, 1.342109202e-10, 1.791623576e-17, 1.518506657e-24,
  0.0004635368241, 1.779561221e-11, 4.038449606e-18, -1.358546052e-25,
  0.0002193961991, 1.779561221e-11, 4.038449606e-18, -1.358546052e-25,
  9.73258866e-05, 1.779561221e-11, 4.038449606e-18, -1.358546052e-25,
  3.62907449e-05, 3.243700447e-12, 5.690024473e-19, 7.09405479e-26,
  5.773168596e-06, 1.424711477e-12, 1.3532163e-19, 1.92417627e-26,
  5.773168596e-06, 1.424711477e-12, 1.3532163e-19, 1.92417627e-26,
  5.773168596e-06, 1.424711477e-12, 1.3532163e-19, 1.92417627e-26,
  1.958472239e-06, 5.152167755e-13, 1.3532163e-19, 1.92417627e-26,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  2.132179588e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  6.420638243e-09, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  6.420638243e-09, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  2.695347945e-09, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  8.327027956e-10, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  8.327027956e-10, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  3.670415083e-10, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  1.342109202e-10, 1.791623576e-17, 1.518506361e-24, 2.613904e-31,
  1.779561221e-11, 4.038449606e-18, -1.358545683e-25, -3.443243946e-32,
  1.779561221e-11, 4.038449606e-18, -1.358545683e-25, -3.443243946e-32,
  1.779561221e-11, 4.038449606e-18, -1.358545683e-25, -3.443243946e-32,
  3.243700447e-12, 5.690024473e-19, 7.094053557e-26, 1.487136711e-32,
  3.243700447e-12, 5.690024473e-19, 7.094053557e-26, 1.487136711e-32,
  3.243700447e-12, 5.690024473e-19, 7.094053557e-26, 1.487136711e-32,
  1.424711477e-12, 1.3532163e-19, 1.924175961e-26, 2.545416018e-33,
  5.152167755e-13, 1.3532163e-19, 1.924175961e-26, 2.545416018e-33,
  6.046956013e-14, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  6.046956013e-14, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  6.046956013e-14, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  1.791623576e-17, 1.518506361e-24, 2.61390353e-31, 4.764937743e-38,
  1.791623576e-17, 1.518506361e-24, 2.61390353e-31, 4.764937743e-38,
  4.038449606e-18, -1.358545683e-25, -3.443243946e-32, 6.296048013e-40,
  4.038449606e-18, -1.358545683e-25, -3.443243946e-32, 6.296048013e-40,
  5.690024473e-19, 7.094053557e-26, 1.487136711e-32, 6.296048013e-40,
  5.690024473e-19, 7.094053557e-26, 1.487136711e-32, 6.296048013e-40,
  5.690024473e-19, 7.094053557e-26, 1.487136711e-32, 6.296048013e-40,
  1.3532163e-19, 1.924175961e-26, 2.545415467e-33, 6.296048013e-40,
  1.3532163e-19, 1.924175961e-26, 2.545415467e-33, 6.296048013e-40,
  2.690143217e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  2.690143217e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  2.690143217e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  1.334890502e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  6.572641438e-21, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  0.05874381959, 1.222115387e-08, 7.693612965e-16, 1.792054435e-22,
  0.02749382704, 4.77057327e-09, 7.693612965e-16, 1.792054435e-22,
  0.01186883077, 1.045283415e-09, 3.252721926e-16, 7.332633139e-23,
  0.00405633077, 1.045283415e-09, 3.252721926e-16, 7.332633139e-23,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  2.801149822e-05, 4.821800945e-12, 8.789757674e-19, 1.208447639e-25,
  2.801149822e-05, 4.821800945e-12, 8.789757674e-19, 1.208447639e-25,
  2.801149822e-05, 4.821800945e-12, 8.789757674e-19, 1.208447639e-25,
  1.275271279e-05, 1.183823005e-12, 1.161414894e-20, 1.291319272e-27,
  5.12331826e-06, 1.183823005e-12, 1.161414894e-20, 1.291319272e-27,
  1.308621904e-06, 2.743283031e-13, 1.161414894e-20, 1.291319272e-27,
  1.308621904e-06, 2.743283031e-13, 1.161414894e-20, 1.291319272e-27,
  3.549478151e-07, 4.695462769e-14, 1.161414894e-20, 1.291319272e-27,
  3.549478151e-07, 4.695462769e-14, 1.161414894e-20, 1.291319272e-27,
  1.165292645e-07, 1.853292503e-14, 4.837885366e-21, 1.291319272e-27,
  1.165292645e-07, 1.853292503e-14, 4.837885366e-21, 1.291319272e-27,
  5.69246339e-08, 4.322073705e-15, 1.449754789e-21, 7.962890365e-29,
  2.712231151e-08, 4.322073705e-15, 1.449754789e-21, 7.962890365e-29,
  1.222115387e-08, 7.693612965e-16, 1.792054182e-22, 2.91418027e-29,
  4.77057327e-09, 7.693612965e-16, 1.792054182e-22, 2.91418027e-29,
  1.045283415e-09, 3.252721926e-16, 7.332632508e-23, 3.898253736e-30,
  1.045283415e-09, 3.252721926e-16, 7.332632508e-23, 3.898253736e-30,
  1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
  1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
  1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
  1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
  5.575349904e-11, 6.083145782e-18, 5.344349223e-25, 1.511644828e-31,
  2.664967552e-11, -8.557475018e-19, -8.595036458e-26, -2.139883875e-32,
  1.209775682e-11, 2.61369883e-18, 5.344349223e-25, 1.511644828e-31,
  4.821800945e-12, 8.789757674e-19, 1.208447639e-25, 3.253064536e-33,
  1.183823005e-12, 1.161414894e-20, 1.29131908e-27, 1.715766248e-34,
  1.183823005e-12, 1.161414894e-20, 1.29131908e-27, 1.715766248e-34,
  2.743283031e-13, 1.161414894e-20, 1.29131908e-27, 1.715766248e-34,
    // clang-format on
}};
#endif // HWY_ONCE
""")


if __name__ == "__main__":
    main()