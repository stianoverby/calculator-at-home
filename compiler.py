#!/usr/bin/env python3

'''
Mini-project: Exploring How To Craft A Compiler

This project involves building a simple compiler for a straightforward language. 
The aim is to delve into the realm of programming language implementation.

Compiler Phases:
    1. Scanning
        - Extract lexemes
        - Classify lexemes into tokens
    2. Parsing
        - Construct an abstract syntax tree

---------------------------------------------------
    Phases Not Covered/Relevant for this project
    ?. Typechecking
    ?. Translation to intermediate representation
    ?. Code optimization
---------------------------------------------------

    4. Translation to Host Assembler
    5. Linking and Translation of Assembler to Executable Binary

The Compiler:
    => Reads files with .calc extension
    => Performs Lexical Analysis
    => Classifies Tokens
    => Parses the Syntax
    => Translates to Host Assembler Language
    => Links Modules and Translates to Executable Code
'''

import sys
import subprocess
import os
import pprint
debug = False

def main():
    if len(sys.argv) < 3:
        usage()
        sys.exit(1)

    global debug
    mode_flag = None
    file_path = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-i' or sys.argv[i] == '-c':
            mode_flag = sys.argv[i]
        elif sys.argv[i] == '-d':
            debug = True
        else:
            file_path = sys.argv[i]
        i += 1

    if mode_flag is None or file_path is None:
        usage()
        sys.exit(1)

    if mode_flag not in ['-i', '-c']:
        usage()
        sys.exit(1)

    if debug:
        print(f"[DEBUG]: Debug mode enabled.")

    if not os.path.exists(file_path):
        if debug:
            print(f"[ERROR]: file does not exists!")
        sys.exit(1)

    if not file_path.endswith('.calc'):
        if debug:
            print(f"[ERROR]: expected .calc file format!")
        sys.exit(1)
    
    if mode_flag == '-i':
        if debug:
            print(f"[DEBUG]: interpreting {file_path}")
        interpret(file_path)
    elif mode_flag == '-c':
        if debug:
            print(f"[DEBUG]: compiling {file_path}")
        compile(file_path)

def usage():
        print(f"Usage: {sys.argv[0]} [-i | -c] program_name.calc")

def interpret(file_path):
    lexemes      = lex_file(file_path)
    tokens       = tokenize(lexemes)
    asts         = parse(tokens)
    evaluate(asts)

def compile(file_path):

    file_name_with_ending = os.path.basename(file_path)
    file_name, _ = file_name_with_ending.split(".")

    lexemes = lex_file(file_path)
    if debug:
        print(f"[DEBUG]: LEXEMES")
        pprint.pprint(lexemes)

    tokens = tokenize(lexemes)
    if debug:
        print(f"[DEBUG]: TOKENS")
        pprint.pprint(tokens)

    ast = parse(tokens)
    if debug:
        print(f"[DEBUG]: ASTS")
        pprint.pprint(ast)

    asm = generate_x86_64_macOS_nasm(ast)

    with open(file_name + ".asm", "w") as f:
        f.writelines(asm)
    
    # Convert asm to object file
    rc = subprocess.call([
        "nasm", 
        "-f", 
        "macho64", 
        f"{file_name}.asm", 
        "-o", f"{file_name}.o"
    ])
    if rc != 0:
        print(f"[ERROR]: not able to compile .asm file to .o file")
        sys.exit(rc)

    # Link object files
    rc = subprocess.call([
        "ld",
        "-macos_version_min",
        "10.13",
        "-e",
        "_start",
        "-static",
        "-o",
        f"{file_name}",
        f"{file_name}.o"
    ])
    if rc != 0:
        print(f"[ERROR]: not able to link .o file")
        sys.exit(rc)

    if not debug: os.remove(f"{file_name}.o")
    if not debug: os.remove(f"{file_name}.asm")

enum_counter = 0
def enum():
    global enum_counter
    enum = enum_counter
    enum_counter += 1
    return enum


TOK_NUM = enum()
TOK_NEWLINE = enum()
TOK_PLUS = enum()
TOK_MINUS = enum()
TOK_MUL = enum()
TOK_DIV = enum()
enum2name = {
    TOK_NUM: "number",
    TOK_NEWLINE: "'\n'",
    TOK_PLUS: "'+'",
    TOK_MINUS: "'-'",
    TOK_MUL: "'*'",
    TOK_DIV: "'/'",
}


#######################
#       LEXER         #
#######################
def lex_file(file_path):
    decorated_lexemes = []
    with open(file_path, "r") as f:
        decorated_lexemes = [
            (file_path, row, col, lexeme)
            for (row, line) in enumerate(f.readlines())
            for (col, lexeme) in lex_line(line)
        ]
    return decorated_lexemes


def lex_line(line):
    lexeme_start = lstrip(line, 0)
    while lexeme_start < len(line):
        lexeme_end = line.find(" ", lexeme_start)
        if lexeme_end < 0:
            lexeme_end = rstrip(line, lexeme_start)
        elif lexeme_end == 0:
            assert False, "Unreachable, lstrip internal error"
        yield (lexeme_start, line[lexeme_start:lexeme_end])
        lexeme_start = lstrip(line, lexeme_end)
    yield (len(line) - 1, "\n")


whitespace = [" ", "\r", "\t", "\n"]
def lstrip(line, col):
    while col < len(line) and line[col] in whitespace:
        col += 1
    return col


def rstrip(line, col):
    while col < len(line) and line[col] not in whitespace:
        col += 1
    return col


#######################
#      TOKENIZER      #
#######################
def tokenize(decorated_lexemes):
    return [tokenize_lexeme(dl) for dl in decorated_lexemes]


def tokenize_lexeme(decorated_lexeme):
    (file_path, row, col, lexeme) = decorated_lexeme
    if lexeme == "+":
        return plus(decorated_lexeme)
    elif lexeme == "-":
        return minus(decorated_lexeme)
    elif lexeme == "*":
        return mul(decorated_lexeme)
    elif lexeme == "/":
        return div(decorated_lexeme)
    elif lexeme == "\n":
        return newline(decorated_lexeme)
    elif lexeme.isdigit():
        return num(decorated_lexeme)
    else:
        print(f"{file_path}:{row}:{col}: {lexeme} is not a valid token")
        exit(1)


def plus(decorated_lexeme):
    (file_path, row, col, _) = decorated_lexeme
    return (file_path, row, col, {"token_type": TOK_PLUS})


def minus(decorated_lexeme):
    (file_path, row, col, _) = decorated_lexeme
    return (file_path, row, col, {"token_type": TOK_MINUS})


def mul(decorated_lexeme):
    (file_path, row, col, _) = decorated_lexeme
    return (file_path, row, col, {"token_type": TOK_MUL})


def div(decorated_lexeme):
    (file_path, row, col, _) = decorated_lexeme
    return (file_path, row, col, {"token_type": TOK_DIV})


def newline(decorated_lexeme):
    (file_path, row, col, _) = decorated_lexeme
    return (file_path, row, col, {"token_type": TOK_NEWLINE})


def num(decorated_lexeme):
    (file_path, row, col, x) = decorated_lexeme
    return (file_path, row, col, {"token_type": TOK_NUM, "value": int(x)})


#######################
#       PARSER        #
#######################
def parse(tokens):
    token_stream = Streamify(tokens)
    AST = program(token_stream)
    eof(token_stream)
    return AST


def program(tokens):
    exprs = []
    if not tokens.peek():
        return exprs  # Empty program
    exprs.append(expression(tokens))
    dtoken = tokens.peek()
    (_, _, _, token) = dtoken
    while token["token_type"] == TOK_NEWLINE:
        tokens.skip()
        if not tokens.peek():
            break  # Reached eof
        exprs.append(expression(tokens))
        dtoken = tokens.peek()
        if not dtoken:
            break
        (_, _, _, token) = dtoken
    return exprs


def expression(tokens):
    left = term(tokens)
    dtoken = tokens.peek()
    if not dtoken:
        return left
    (_, _, _, token) = dtoken
    while token["token_type"] == TOK_PLUS or token["token_type"] == TOK_MINUS:
        tokens.skip()
        right = term(tokens)
        left = {"op": token, "left": left, "right": right}
        dtoken = tokens.peek()
        if not dtoken:
            break
        (_, _, _, token) = dtoken
    return left


def term(tokens):
    left = literal(tokens)
    dtoken = tokens.peek()
    if not dtoken:
        return left

    (_, _, _, token) = dtoken
    while token["token_type"] == TOK_MUL or token["token_type"] == TOK_DIV:
        tokens.skip()
        right = literal(tokens)
        left = {"op": token, "left": left, "right": right}
        dtoken = tokens.peek()
        if not dtoken:
            break
        (_, _, _, token) = dtoken
    return left


def literal(tokens):
    dtoken = tokens.next()
    if not dtoken:
        parserError("", -1, -1, f"expected number, but got eof")
    (file_path, row, col, token) = dtoken
    if token["token_type"] != TOK_NUM:
        parserError(
            file_path,
            row,
            col,
            f"expected number, but got {enum2name[token["token_type"]]}",
        )
    return token


def eof(tokens):
    dtok = tokens.next()
    if dtok:
        (file_path, row, col, token) = dtok
        parserError(
            file_path, row, col, f"expected eof, but got {enum2name[token["token_type"]]}"
        )



def parserError(file_path, row, col, msg):
    print(f"{file_path}:{row}:{col}: Parser Error: {msg}")
    sys.exit(1)


class Streamify(object):
    def __init__(self, iterator):
        self.head = None
        self.iterator = iter(iterator)
        self.peeked = False

    def peek(self):
        if not self.peeked:
            self.peeked = True
            self.head = next(self.iterator, None)
        return self.head

    def next(self):
        if self.peeked:
            self.peeked = False
            return self.head
        else:
            return next(self.iterator, None)

    def skip(self):
        if self.peeked:
            self.peeked = False
            return
        else:
            next(self.iterator, None)

    def __next__(self):
        if self.peeked:
            self.peeked = False
            return self.head
        else:
            return next(self.iterator, None)


#######################
#     INTERPRETER     #
#######################
def evaluate(asts):
    evalutated_terms = [eval(ast) for ast in asts]
    for et in evalutated_terms:
        print(et)


def eval(ast):
    if "op" in ast:
        op_type = ast["op"]["token_type"]
        if op_type == TOK_PLUS:
            return eval_plus(ast["left"], ast["right"])
        elif op_type == TOK_MINUS:
            return eval_minus(ast["left"], ast["right"])
        elif op_type == TOK_MUL:
            return eval_mul(ast["left"], ast["right"])
        elif op_type == TOK_DIV:
            return eval_div(ast["left"], ast["right"])
    assert ast["token_type"] == TOK_NUM
    return eval_num(ast)


def eval_plus(left, right):
    return eval(left) + eval(right)


def eval_minus(left, right):
    return eval(left) - eval(right)


def eval_mul(left, right):
    return eval(left) * eval(right)


def eval_div(left, right):
    return eval(left) / eval(right)


def eval_num(ast):
    return ast["value"]

#######################
#    ASM GENERATOR    #
#######################
def generate_x86_64_macOS_nasm(asts):
    nasm = []
    for ast in asts:
        nasm += ast_2_nasm(ast) + asm_call_print_number()
    return (
        asm_header()        + 
        asm_stdlib()        + 
        asm_entrypoint()    + 
        nasm    + 
        asm_syscall_exit()
    )
    

def ast_2_nasm(ast):
    asm_code = []
    if "op" in ast:
        asm_code += ast_2_nasm(ast["left"])
        asm_code += ast_2_nasm(ast["right"])
        op_type = ast["op"]["token_type"]
        if op_type == TOK_PLUS:
            asm_code += asm_plus()
        elif op_type == TOK_MINUS:
            asm_code += asm_minus()
        elif op_type == TOK_MUL:
            asm_code += asm_mul()
        elif op_type == TOK_DIV:
            asm_code += asm_div()
    else:
        assert ast["token_type"] == TOK_NUM
        asm_code += asm_num(ast)
    
    return asm_code

def asm_plus():
    asm_code = []
    asm_code.append("    ;; -- Addition -- ;;\n")
    asm_code.append("    pop rax\n"             ) # Pop right operand into rax
    asm_code.append("    pop rbx\n"             ) # Pop left operand into rbx
    asm_code.append("    add rbx, rax\n"        ) # Add right operand to left operand
    asm_code.append("    push rbx\n"            ) # Push the result back onto the stack 
    return asm_code

def asm_minus():
    asm_code = []
    asm_code.append("    ;; -- Subtraction -- ;;\n")
    asm_code.append("    pop rax\n"                ) # Pop right operand into rax
    asm_code.append("    pop rbx\n"                ) # Pop left operand into rbx
    asm_code.append("    sub rbx, rax\n"           ) # Subtract right operand from left operand
    asm_code.append("    push rbx\n"               ) # Push the result back onto the stack 
    return asm_code

def asm_mul():
    asm_code = []
    asm_code.append("    ;; -- Multiplication -- ;;\n")
    asm_code.append("    pop rax\n"                   ) # Pop right operand into rax
    asm_code.append("    pop rbx\n"                   ) # Pop left operand into rbx
    asm_code.append("    imul rbx, rax\n"             ) # Multiply right operand by left operand
    asm_code.append("    push rbx\n"                  ) # Push the result back onto the stack
    return asm_code

def asm_div():
    asm_code = []
    asm_code.append("    ;; -- Division -- ;;\n")
    asm_code.append("    pop rbx\n"             ) # Pop dividend into rbx
    asm_code.append("    pop rax\n"             ) # Pop divisor into rax
    asm_code.append("    cqo\n"                 ) # Sign-extend rax into rdx:rax for signed division
    asm_code.append("    idiv rbx\n"            ) # Divide rbx:rax by rbx, quotient in rax, remainder in rdx
    asm_code.append("    push rax\n"            ) # Push the quotient back onto the stack

    return asm_code

def asm_num(ast):
    asm_code = [f"    push {ast["value"]}\n"]
    return asm_code

def asm_call_print_number():
    asm_code = []
    asm_code.append("    ;; -- Call print_number -- ;;\n")
    asm_code.append("    pop rdi\n"                      )
    asm_code.append("    call print_number\n"            )
    return asm_code

def asm_syscall_exit():
    asm_code = []
    asm_code.append("    ;; -- Call exit(0) -- ;;\n")
    asm_code.append("    mov rax, 0x2000001\n"      )
    asm_code.append("    xor rdi, rdi\n"            )
    asm_code.append("    syscall\n"                 )
    return asm_code

def asm_header():
    return [
        "BITS 64\n",
        "    global _start:\n",
        "    section .text\n"
        ]

def asm_stdlib():
    asm_code = [ 
        ";; -- stdlib -- ;;\n",
        "print_number:\n",
        "    sub     rsp, 40\n",
        "    mov     rcx, rdi\n",
        "    xor     r10d, r10d\n",
        "    test    rdi, rdi\n",
        "    jns     .L2\n",
        "    neg     rcx\n",
        "    mov     r10d, 1\n",
        ".L2:\n",
        "    mov     BYTE [rsp+31], 10\n",
        "    mov     esi, 1\n",
        "    lea     r9, [rsp+31]\n",
        "    mov     r8, 7378697629483820647\n",
        ".L3:\n",
        "    mov     rax, rcx\n",
        "    mov     rdi, r9\n",
        "    imul    r8\n",
        "    mov     rax, rcx\n",
        "    sub     rdi, rsi\n",
        "    sar     rax, 63\n",
        "    sar     rdx, 2\n",
        "    sub     rdx, rax\n",
        "    lea     rax, [rdx+rdx*4]\n",
        "    add     rax, rax\n",
        "    sub     rcx, rax\n",
        "    mov     rax, rsi\n",
        "    add     rsi, 1\n",
        "    add     ecx, 48\n",
        "    mov     BYTE [rdi], cl\n",
        "    mov     rcx, rdx\n",
        "    test    rdx, rdx\n",
        "    jne     .L3\n",
        "    test    r10d, r10d\n",
        "    je      .L4\n",
        "    not     rsi\n",
        "    mov     BYTE [rsp+32+rsi], 45\n",
        "    lea     rsi, [rax+2]\n",
        ".L4:\n",
        "    mov     eax, 32\n",
        "    mov     rdx, rsi\n",
        "    mov     edi, 1\n",
        "    sub     rax, rsi\n",
        "    add     rax, rsp\n",
        "    mov     rsi, rax\n",
        "    mov rax, 0x2000004\n",
	    "    mov rdi, 1\n",
        "    syscall\n",
        "    add     rsp, 40\n",
        "    ret\n"]
    return asm_code

def asm_entrypoint():
    return [
        "_start:\n"
    ]

if __name__ == '__main__':
    main()
