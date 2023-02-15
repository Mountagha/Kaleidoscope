#include "../include/KaleidoscopeJIT.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InsCombine/InsCombine.h"
#include "llvm/Transform/Scalar.h"
#include "llvm/Transform/Scalar/GVN.h"

#include <algorithm>
#include <string>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

//===--------------------------------------------------------------------------===//
// lexer
//===--------------------------------------------------------------------------===//

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one 
// of these for known things.

enum Token {
    tok_eof = -1,

    // commands
    tok_def = -2,
    tok_extern = -3,

    // primary
    tok_identifier = -4,
    tok_number = -5
};

static std::string IdentifierStr; // Filled in if tok_identifier
static double NumVal;   // Filled in if tok_number

// gettok - Return the next token from standard input.
static int gettok() {
    static int LastChar = ' ';

    // Skip any whitespace.
    while(isspace(LastChar))
        LastChar = getchar();
    
    if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
        IdentifierStr = LastChar;
        while (isalnum((LastChar = getchar())))
            IdentifierStr += LastChar;

        if (IdentifierStr == "def")
            return tok_def;
        if(IdentifierStr == "extern")
            return tok_extern;
        return tok_identifier;
    }
    if (isdigit(LastChar)) {
        // Number: [0-9]+
        std::string NumStr;
        do {
            NumStr += LastChar;
            LastChar = getchar();
        } while (isdigit(LastChar));

        // look for a fractional part.
        if (LastChar == '.')
            NumStr += LastChar;

        // consume the decimal part.
        while(isdigit(LastChar)) {
            NumStr += LastChar;
            LastChar = getchar();
        }
        NumVal = strtod(NumStr.c_str(),  nullptr);
        return tok_number;
    }

    if (LastChar == '#') {
        // comment until the end of the line.
        do 
            LastChar = getchar();
        while(LastChar != EOF && LastChar != '\n' && LastChar != '\r');
        if (LastChar != EOF) return gettok();
    }
    // Check for the end of file. Don't eat the EOF.
    if (LastChar == EOF)
        return tok_eof;
    // Otherwise, just return the character as its ascii value
    int thisChar = LastChar;
    LastChar = getchar();
    return thisChar;
}

//===--------------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===--------------------------------------------------------------------------===//

namespace {
/// Express AST - Base class for all expressions nodes.
class ExprAST {
    public:
        virtual ~ExprAST() = default;
        virtual Value* codegen() = 0;
};

/// NumberExprAST - Expression class for numeric literals like "1.0"
class NumberExprAST : public ExprAST {
    double Val;
    public:
        NumberExprAST(double val): Val(val) {}
        Value* codegen() override;
};

/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
    std::string Name;

    public:
        VariableExprAST(const std::string &name) : Name(name) {}
        Value* codegen() override;
};

/// BinaryExprAST - Expression class for functions calls.
class BinaryExprAST : public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;
    public:
        BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs, 
                    std::unique_ptr<ExprAST> rhs)
        : Op(op), LHS(std::move(lhs)), RHS(std::move(rhs)) {}
        Value* codegen() override;
};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;
    public:
        CallExprAST(const std::string &callee, std::vector<std::unique_ptr<ExprAST>> args)
        : Callee(callee), Args(std::move(args)) {}
        Value* codegen() override;
};

/// PrototypeAST - This class represents the "prototype" for a function
/// which captures its name, and its argument names (thus implicitly the number of arguments 
/// the function takes)

class PrototypeAST {
    std::string Name;
    std::vector<std::string> Args;

    public:
        PrototypeAST(const std::string &name, std::vector<std::string> Args)
            : Name(name), Args(std::move(Args)) {}

        Function* codegen(); 
        const std::string &getName() const { return Name; }
};

/// FunctionAST - this class represents a function defintion itself.
class FunctionAST {
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprAST> Body;

    public:
        FunctionAST(std::unique_ptr<PrototypeAST> proto, 
                    std::unique_ptr<ExprAST> body)
            : Proto(std::move(proto)), Body(std::move(body)) {}
        Function* codegen();
};

} // End of anonymous namespace.

//===--------------------------------------------------------------------------===//
// Parser 
//===--------------------------------------------------------------------------===//
/// CurTok/getNextToken - Provide a simple token buffer. Curtok is the current
/// token the parser is looking at. getNextToken reads another token from the 
/// lexer and updates curTok with its results.

static int curTok;
static int getNextToken() {
    return curTok = gettok();
}

/// BinopPrecedence - This holds the precedence for each binary operator that is defined
static std::map<char, int> binopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int getTokPrecedence() {
    if (!isascii(curTok)) {
        return -1;
    }

    // make sure it's a declared binop
    int tokPrec = binopPrecedence[curTok];
    if (tokPrec <= 0) return -1;
    return tokPrec;
}

// / * - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogError(const char* str) {
    fprintf(stderr, "LogError: %s\n", str);
    return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char* str) {
    LogError(str);
    return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

/// number ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
    auto result = std::make_unique<NumberExprAST>(NumVal);
    getNextToken(); // consume the next token
    return std::move(result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
    getNextToken();// eat (.
    auto V = ParseExpression();
    if (!V) {
        return nullptr;
    }
    if (curTok != ')') 
        return LogError("expected ')'. ");
    getNextToken();
    return V;

}

/// Identifierexpr
///  ::= identifier
///  ;:= identifier '(' identifier ')'

static std::unique_ptr<ExprAST> parseIdentifierExpr() {
    std::string idName = IdentifierStr;

    getNextToken();     // eat identifier   

    if (curTok == '(') // Simple variable ref.
        return std::make_unique<VariableExprAST> (idName);

    // Call.
    getNextToken();  // eat.
    std::vector<std::unique_ptr<ExprAST>> args;
    if (curTok != ')') {
        while (1) {
            if (auto Arg = ParseExpression() )
                args.push_back(std::move(Arg));
            else 
                return nullptr;
            if (curTok == ')')
                break;
            if (curTok != ',')
                return LogError("Expected ) or , in argument list.");
            getNextToken();
        }
    }
    getNextToken();
    return std::make_unique<CallExprAST> (idName, std::move(args));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
  switch (curTok) {
  default:
    return LogError("unknown token when expecting an expression");
  case tok_identifier:
    return parseIdentifierExpr();
  case tok_number:
    return ParseNumberExpr();
  case '(':
    return ParseParenExpr();
  }
}



/// binorphs '
/// ::= ('+' primary)*
static std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec, std::unique_ptr<ExprAST> lhs) {
    // If this is a binop, find its precedence.
    while (true) {
        int tokPrec = getTokPrecedence();

        // If this a binop that binds at least as tightly as the current binop,
        // consume it, otherwise we are done.
        if (tokPrec < exprPrec) return lhs;
    
        // okay, we know this is a binop.
        int binOps = curTok;
        getNextToken(); // eat binop

        // Parse the primary expression after the binary operator.
        auto rhs = ParsePrimary();
        if (!rhs) return nullptr;

        // If binOp binds less tightly with RHS than the operator after RHS, let
        // the pending operator take RHS as LHS.
        int nextPrec = getTokPrecedence();
        if (tokPrec < nextPrec) {
            rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
            if (!rhs) return nullptr;
        }
        
        // Merge LHS/RHS
        lhs = std::make_unique<BinaryExprAST>(binOps, std::move(lhs), std::move(rhs));
    }
    
}

/// expression 
/// ::= primary binoprhs
static std::unique_ptr<ExprAST> ParseExpression() {
    auto lhs = ParsePrimary();
    if (!lhs) return nullptr;
    return parseBinOpRHS(0, std::move(lhs));
}

/// prototype
/// ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> parsePrototype() {
    if (curTok != tok_identifier)
        LogErrorP("Expected function name in prototype");

    std::string fnName = IdentifierStr;
    getNextToken();

    if (curTok != '(')
        return LogErrorP("Expected '(' in prototype");
    
    std::vector<std::string> argNames;
    while (getNextToken() == tok_identifier) {
        argNames.push_back(IdentifierStr);
    }
    if (curTok != ')')
        return LogErrorP("Expected ')' in prototype");

    // success.
    getNextToken(); // eat ')'.
    return std::make_unique<PrototypeAST>(fnName, std::move(argNames));
}

/// defintion ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> parseDefintion() {
    getNextToken(); // eat def.
    auto proto = parsePrototype();
    if (!proto) return nullptr;
    if (auto E = ParseExpression()) 
        return std::make_unique<FunctionAST>(std::move(proto), std::move(E));
    return nullptr;
}

// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> parseTopLevelExpr() {
    if (auto E = ParseExpression()) {
        // Make an anonymous proto.
        auto proto = std::make_unique<PrototypeAST>("__anon_expr", std::vector<std::string>());
        return std::make_unique<FunctionAST>(std::move(proto), std::move(E));
    }
    return nullptr;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> parseExtern() {
    getNextToken();// eat extern.
    return parsePrototype();
} 

//==--------------------------------------------------------------
// Code generation 
//==--------------------------------------------------------------

static std::unique_ptr<LLVMContext> theContext;
static std::unique_ptr<Module> theModule;
static std::unique_ptr<IRBuilder<>> builder;
static std::map<std::string, Value* > namedValues;
static std::unique_ptr<Legacy::FunctionPassManager> TheFPM;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static ExitOnError ExitOnErr;

Value* LogErrorV(const char *Str) {
    LogError(Str);
    return nullptr;
}

Value* NumberExprAST::codegen() {
    return ConstantFP::get(*theContext, APFloat(Val));
}

Value* VariableExprAST::codegen() {
    // Look this variable up in the function.
    Value* V = namedValues[Name];
    if (!V)
        return LogErrorV("Unknown variable name.");
    return V;
}

Value* BinaryExprAST::codegen() {
    Value* L = LHS->codegen();
    Value* R = RHS->codegen();
    if (!L || !R) 
        return nullptr;
    switch(Op) {
        case '+':
            return builder->CreateFAdd(L, R, "addtmp");
        case '-':
            return builder->CreateFSub(L, R, "subtmp");
        case '*':
            return builder->CreateFMul(L, R, "multmp");
        case '<':
            L = builder->CreateFCmpULT(L, R, "comtmp");
            // Convert bool 0/1 to double 0.0 or 1.0
            return builder->CreateUIToFP(L, Type::getDoubleTy(*theContext), "booltmp");
        default:
            LogErrorV("Invalid binary operator");
    }
}

Value* CallExprAST::codegen() {
    // Look up the name in the global module table.
    Function *CalleeF = theModule->getFunction(Callee);
    if (!CalleeF) 
        return LogErrorV("unknown function referenced.");
    // If argument mismatch erro.
    if (CalleeF->arg_size() != Args.size()) 
        return LogErrorV("Incorrect # of arguments passed.");
    std::vector<Value*> argsV;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
        argsV.push_back(Args[i]->codegen());
        if (!Args.back())
            return nullptr;
    }

    return builder->CreateCall(CalleeF, argsV, "calltmp");
}

Function* PrototypeAST::codegen() {
    // Make the function type: double(double, double) etc.
    std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(*theContext));
    FunctionType* FT = FunctionType::get(Type::getDoubleTy(*theContext), Doubles, false);
    Function* F = Function::Create(FT, Function::ExternalLinkage, Name, theModule.get());

    // Set names for all arguments.
    unsigned idx = 0;
    for (auto &Arg : F->args()) {
        Arg.setName(Args[idx++]);
    }
    return F;
}

Function *FunctionAST::codegen() {
    // First, check for an existing function from a previous 'extern' declaration.
    Function* TheFunction = theModule->getFunction(Proto->getName());

    if (!TheFunction)
        TheFunction = Proto->codegen();
    if (!TheFunction)
        return nullptr;
    
    // Create a new basic block to start insertion into.
    BasicBlock *BB = BasicBlock::Create(*theContext, "entry", TheFunction);

    // Record the function arguments in the NamedValues map.
    namedValues.clear();
    for (auto &Arg: TheFunction->args())
        namedValues[std::string(Arg.getName())] = &Arg;
    
    if (Value* RetVal = Body->codegen()) {
        // finish off the function.
        builder->CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        return TheFunction;
    }

    // Error reading body, remove function.
    TheFunction->eraseFromParent();
    return nullptr;
}
//==--------------------------------------------------------------
// Top-level parsing
//==--------------------------------------------------------------

static void InitializeModule() {
    // Open a new context and module.
    theContext = std::make_unique<LLVMContext>();
    theModule = std::make_unique<Module>("my cool jit", *theContext);

    // Create a new builder for the module.
    builder = std::make_unique<IRBuilder<>>(*theContext);
}

static void handleDefinition() {
    if(parseDefintion()) {
        fprintf(stderr, "Parsed a function definition.\n");
    } else {
        // skip token for error recovery.
        getNextToken();
    }
}

static void handleExtern() {
    if (parseExtern()) {
        fprintf(stderr, "Parsed an extern.\n");
    } else {
        // skip token for error recovery.
        getNextToken();
    }
}

static void handleTopLevelExpression() {
    if (parseTopLevelExpr()) {
        fprintf(stderr, "Parsed a top-level expr.\n");
    } else {
        // skip token for error recovery
        getNextToken();
    }
}

/// top ::= definition | external | expression | ';'
static void mainLoop() {
    while (true) {
        fprintf(stderr, "ready> ");
        switch (curTok) {
            case tok_eof:
                return;
            case ';':   // ignore top-level semicolons.
                getNextToken();
                break;
            case tok_def:
                handleDefinition();
                break;
            case tok_extern:
                handleExtern();
                break;
            default:
                handleTopLevelExpression();
                break;
        }
    }
}

//==--------------------------------------------------------------
// Main driver code.
//==--------------------------------------------------------------


int main() {
    // install standard binary operators.
    // 1 is lowest precedence.
    binopPrecedence['<'] = 10;
    binopPrecedence['+'] = 20;
    binopPrecedence['-'] = 30;
    binopPrecedence['*'] = 40;  // highest.

    // Prime the first token.
    fprintf(stderr, "ready> ");
    getNextToken();

    // Make the module, which holds all the code.
    InitializeModule();

    // Run the main 'Interpreter loop" now.
    mainLoop();

    // Print out all of the generated code.
    theModule->print(errs(), nullptr);

    return 0;
}
