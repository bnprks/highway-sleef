
import collections

from tree_sitter import Language, Parser
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compile treesitter grammars")
    parser.add_argument("src_dir", help="Folder containing tree-sitter grammar repos")
    args = parser.parse_args()

__lib_path = None
def lib_path():
    global __lib_path
    if __lib_path is None:
        raise Exception("Must call set_treesitter_lib() prior to lib_path()")
    else:
        return __lib_path

def c_query(query):
    return Language(tsc.language()).query(query)

def cpp_query(query):
    return Language(tscpp.language()).query(query)

def c_parse(text):
    parser = Parser()
    parser.set_language(Language(tsc.language()))
    return parser.parse(text)

def cpp_parse(text):
    parser = Parser()
    parser.set_language(Language(tscpp.language()))
    return parser.parse(text)

def construct_callgraph(root_node):
    query = c_query(
    """
    (call_expression function: (identifier) @match)
    """)
    calls = collections.defaultdict(set)
    for n, _ in query.captures(root_node):
        parent = n.parent
        # Beware some text like __attribute__((weak, alias(str_xacosf_u1  ) which gets alias parsed as a top-level function call
        while parent.type != "function_definition":
            if parent.type == "attribute_specifier":
                break
            parent = parent.parent
        if parent.type != "function_definition":
            break
        caller_fn = parent.child_by_field_name("declarator").child_by_field_name("declarator").text.decode()
        
        calls[caller_fn].add(n.text.decode())
    return calls

def make_transitive_callgraph(callgraph):
    """Get map of all transitive calls from a callgraph

    Args:
        callgraph (dict[str, set[str]]): Map from a function name to its direct callees
    
    Return: Map from a function name to all transitive callees in the callgraph
    """
    orig_callgraph = callgraph
    while True:
        new_callgraph = {}
        for parent, children in callgraph.items():
            new_callgraph[parent] = set(children)
            for child in children:
                if child in orig_callgraph:
                    new_callgraph[parent] |= orig_callgraph[child]
        
        if new_callgraph == callgraph:
            break
        callgraph = new_callgraph
    return callgraph


def all_defined_functions(root_node):
    query = c_query(
    """
    (function_definition 
        declarator: (function_declarator
            declarator: (identifier) @match))
    """)
    fns = set()
    for n, _ in query.captures(root_node):
        fns.add(n.text.decode())
    return fns

if __name__ == "__main__":
    main()