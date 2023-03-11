#include "include/KaleidoscopeJIT.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
//#include "llvm/IR/Instructions"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

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
    tok_number = -5,
    
    // control
    tok_if = -6,
    tok_then = -7,
    tok_else = -8,
    tok_for = -8,
    tok_in = -10,
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
        if (IdentifierStr == "if")
            return tok_if;
        if (IdentifierStr == "then")
            return tok_then;
        if (IdentifierStr == "else")
            return tok_else;
        if (IdentifierStr == "for")
            return tok_for;
        if (IdentifierStr == "in")
            return tok_in;
        return tok_identifier;

    }
    if (isdigit(LastChar) || LastChar == '.') {
        // Number: [0-9.]+
        std::string NumStr;
        do {
            NumStr += LastChar;
            LastChar = getchar();
        } while (isdigit(LastChar) || LastChar == '.');

        // look for a fractional part.
        //if (LastChar == '.')
        //    NumStr += LastChar;

        // consume the decimal part.
        //while(isdigit(LastChar)) {
        //    NumStr += LastChar;
        //    LastChar = getchar();
        //}
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

/// IfExprAST - Expression class for if/then/else.
class IfExprAST : public ExprAST {
    std::unique_ptr<ExprAST> Cond, Then, Else;

public:
    IfExprAST(std::unique_ptr<ExprAST> cond, std::unique_ptr<ExprAST> then, 
            std::unique_ptr<ExprAST> else_)
        : Cond(std::move(cond)), Then(std::move(then)), Else(std::move(else_)) {}

    Value* codegen() override;
};

/// ForExprAST - Expression class for for/in.
class ForExprAST : public ExprAST {
    std::string VarName;
    std::unique_ptr<ExprAST> Start, End, Step, Body;

public:
    ForExprAST(const std::string& varName, std::unique_ptr<ExprAST> start,
            std::unique_ptr<ExprAST> end, std::unique_ptr<ExprAST> step, 
            std::unique_ptr<ExprAST> body)
        : VarName(varName), Start(std::move(start)), End(std::move(end)),
          Step(std::move(step)), Body(std::move(body)) {}
        
    Value* codegen() override;
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

    if (curTok != '(') // Simple variable ref.
        return std::make_unique<VariableExprAST> (idName);

    // Call.
    getNextToken();  // eat (.
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
    // Eat the ')'.
    getNextToken();
    return std::make_unique<CallExprAST> (idName, std::move(args));
}

/// ifexpr ::= 'if' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr() {
    getNextToken();  // eat the if.

    // condition.
    auto Cond = ParseExpression();
    if (!Cond)
        return nullptr;
    if (curTok != tok_then) 
        return LogError("Expected 'then'.");
    getNextToken();  // eat the then.

    auto Then = ParseExpression();
    if (!Then)
        return nullptr;
    if (curTok != tok_else) 
        return LogError("Expected 'else'.");
    
    getNextToken();

    auto Else = ParseExpression();
    if (!Else)
        return nullptr;
    return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then), std::move(Else));
}

/// forexpr ::= 'for' identifier '=' expr ',' expr (',', expr) ? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr() {
    getNextToken();     // eat the for.

    if (curTok != tok_identifier)
        return LogError("expected identifier after for");
    std::string IdName = IdentifierStr;
    getNextToken(); // eat identifier.

    if (curTok != '=')
        return LogError("expected '=' after for");
    getNextToken(); // eat '='.

    auto Start = ParseExpression();
    if (Start)
        return nullptr;
    if (curTok != ',')
        return LogError("expected ',' after for start value");
    getNextToken();

    auto End = ParseExpression();
    if (!End)
        return nullptr;
    
    // The step value is optional.
    std::unique_ptr<ExprAST> Step;
    if (curTok == ',') {
        getNextToken();
        Step = ParseExpression();
        if (!Step)
            return nullptr;
    }

    if (curTok != tok_in)
        return LogError("expected 'in' after for");
    getNextToken(); // eat 'in'.

    auto Body = ParseExpression();
    if (!Body)
        return nullptr;
    
    return std::make_unique<ForExprAST>(IdName, std::move(Start),
                                        std::move(End), std::move(Step),
                                        std::move(Body));
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
  case tok_if:
    return ParseIfExpr();
  case tok_for:
    return ParseForExpr();
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
        int binOp = curTok;
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
        lhs = std::make_unique<BinaryExprAST>(binOp, std::move(lhs), std::move(rhs));
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
static std::unique_ptr<legacy::FunctionPassManager> TheFPM;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static ExitOnError ExitOnErr;

Value* LogErrorV(const char *Str) {
    LogError(Str);
    return nullptr;
}

Function *getFunction(std::string name) {
    // First, see if the function has already been added to the currrent module.
    if (auto* F = theModule->getFunction(name))
        return F;
    // If not, check whether we can codegen the declaration from some existing 
    // prototype.
    auto FI = FunctionProtos.find(name);
    if (FI != FunctionProtos.end())
        return FI->second->codegen();

    // if no existing prototype exists, return null.
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
            return LogErrorV("Invalid binary operator");
    }
}

Value* CallExprAST::codegen() {
    // Look up the name in the global module table.
    Function *CalleeF = getFunction(Callee);
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
    // Transfer ownership of the prototype to the FunctionProtos map, but keep a 
    // reference to it for use below.
    auto &P = *Proto;
    FunctionProtos[Proto->getName()] = std::move(Proto);
    Function *TheFunction = getFunction(P.getName());
    if (!TheFunction)
        return nullptr;

    // Create a new basic block to start insertion into.
    BasicBlock *BB = BasicBlock::Create(*theContext, "entry", TheFunction);
    builder->SetInsertPoint(BB);

    // Record the function arguments in the NamedValues map.
    namedValues.clear();
    for (auto &Arg: TheFunction->args())
        namedValues[std::string(Arg.getName())] = &Arg;
    
    if (Value* RetVal = Body->codegen()) {
        // finish off the function.
        builder->CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        // Run the optimizer on the function.
        TheFPM->run(*TheFunction);

        return TheFunction;
    }

    // Error reading body, remove function.
    TheFunction->eraseFromParent();
    return nullptr;
}

Value* IfExprAST::codegen() {
    Value* CondV = Cond->codegen();
    if (!CondV)
        return nullptr;
    
    // Convert condition to a bool by comparing non-equal to 0.0
    CondV = builder->CreateFCmpONE(
        CondV, ConstantFP::get(*theContext, APFloat(0.0)), "ifcond");

    Function *TheFunction = builder->GetInsertBlock()->getParent();

    // Create blocks for the then and else cases. Insert the 'then' block at the
    // end of the function.
    BasicBlock *ThenBB = 
        BasicBlock::Create(*theContext, "then", TheFunction);
    BasicBlock *ElseBB = BasicBlock::Create(*theContext, "else");
    BasicBlock *MergeBB = BasicBlock::Create(*theContext, "ifcont");

    builder->CreateCondBr(CondV, ThenBB, ElseBB);

    // Emit then value.
    builder->SetInsertPoint(ThenBB);

    Value *ThenV = Then->codegen();
    if (!ThenV)
        return nullptr;
    
    builder->CreateBr(MergeBB);
    // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
    ThenBB = builder->GetInsertBlock();

    // Emit the else block.
    TheFunction->insert(TheFunction->end(), ElseBB);
    builder->SetInsertPoint(ElseBB);
    
    Value* ElseV = Else->codegen();
    if (!ElseV) 
        return nullptr;
    
    builder->CreateBr(MergeBB);
    // codegen of 'Else' can change the current block, update ElseBB for the PHI.
    ElseBB = builder->GetInsertBlock();

    // Emit merge block.
    TheFunction->insert(TheFunction->end(), MergeBB);
    builder->SetInsertPoint(MergeBB);
    PHINode *PN = 
        builder->CreatePHI(Type::getDoubleTy(*theContext), 2, "iftmp");
    PN->addIncoming(ThenV, ThenBB);
    PN->addIncoming(ElseV, ElseBB);
    return PN;

}

Value* ForExprAST::codegen() {
    // Emit the start code first, without 'variable' in scope.
    Value* StartVal = Start->codegen();
    if (!StartVal)
        return nullptr;
    
    // Make the new basic block for the loop header, inserting after current block.
    Function* TheFunction = builder->GetInsertBlock()->getParent();
    BasicBlock *PreHeaderBB = builder->GetInsertBlock();
    BasicBlock *LoopBB = BasicBlock::Create(*theContext, "loop", TheFunction);

    // Insert an explicit fall through from the current block to the LoopBB.
    builder->CreateBr(LoopBB);

    // Start insertion in LoopBB
    builder->SetInsertPoint(LoopBB);

    // Start the PHI node with an entry for Start.
    PHINode *Variable = builder->CreatePHI(Type::getDoubleTy(*theContext), 2, VarName);
    Variable->addIncoming(StartVal, PreHeaderBB);

    // Within the loop, the variable is defined equal to the PHI node. If it 
    // shadows an existing variable, we have to restore it, so save it now.
    Value* OldVal = namedValues[VarName];
    namedValues[VarName] = Variable;

    // Emit the body of the loop. This, like any other expr, can change the
    // current BB. Note that we ignore the value computed by the body, but don't 
    // allow an error.
    if(!Body->codegen())
        return nullptr;

    // Emit the step value.
    Value* StepVal = nullptr;
    if (Step) {
        StepVal = Step->codegen();
        if (!StepVal)
            return nullptr;
    } else {
        // If not specified, use 1.0.
        StepVal = ConstantFP::get(*theContext, APFloat(1.0));
    }

    Value* NextVar = builder->CreateFAdd(Variable, StepVal, "nextvar");

    // Compute the end condition.
    Value *EndCond = End->codegen();
    if (!EndCond) 
        return nullptr;
    
    // Convert condition to a bool by comparing non-equal to 0.0.
    EndCond = builder->CreateFCmpONE(
        EndCond, ConstantFP::get(*theContext, APFloat(0.0)), "loopcond"
    );

    // Create the "after loop" block and insert it.
    BasicBlock *LoopEndBB = builder->GetInsertBlock();
    BasicBlock *AfterBB = BasicBlock::Create(*theContext, "afterloop", TheFunction);

    // Insert the conditional branch into the end of LoopEndBB.
    builder->CreateCondBr(EndCond, LoopBB, AfterBB);

    // Any new code will be inserted in AfterBB.
    builder->SetInsertPoint(AfterBB);

    // Add a new entry to the PHI node for the backedge.
    Variable->addIncoming(NextVar, LoopEndBB);

    // Restore the unshadowed variable.
    if (OldVal)
        namedValues[VarName] = OldVal;
    else
        namedValues.erase(VarName);
    
    // for expr always return 0.0.
    return Constant::getNullValue(Type::getDoubleTy(*theContext));
}
//==--------------------------------------------------------------
// Top-level parsing and JIT Driver
//==--------------------------------------------------------------

static void InitializeModuleAndPassManager() {
    // Open a new context and module.
    theContext = std::make_unique<LLVMContext>();
    theModule = std::make_unique<Module>("my cool jit", *theContext);
    theModule->setDataLayout(TheJIT->getDataLayout());

    // Create a new builder for the module.
    builder = std::make_unique<IRBuilder<>>(*theContext);

    // Create a new pass manager attached to it.
    TheFPM = std::make_unique<legacy::FunctionPassManager>(theModule.get());

    // Do simple "peephole" optimization and bit-twiddling optzns.
    TheFPM->add(createInstructionCombiningPass());
    // Reassociate expressions.
    TheFPM->add(createReassociatePass());
    // Eliminate Common SubExpressions.
    TheFPM->add(createGVNPass());
    // Simplify the control flow graph (deleting unreachable blocks, etc).
    TheFPM->add(createCFGSimplificationPass());

    TheFPM->doInitialization();
    
}

static void handleDefinition() {
    if (auto FnAST = parseDefintion()) {
        if (auto *FnIR = FnAST->codegen()) {
            fprintf(stderr, "Read function definition:\n");
            FnIR->print(errs());
            fprintf(stderr, "\n");
            ExitOnErr(TheJIT->addModule(
                ThreadSafeModule(std::move(theModule), std::move(theContext))
            ));
            InitializeModuleAndPassManager();
        }
    } else {
        // skip token for error recovery.
        getNextToken();
    }
}

static void handleExtern() {
    if (auto ProtoAST = parseExtern()) {
        if (auto *FnIR = ProtoAST->codegen()) {
            fprintf(stderr, "Read extern:\n");
            FnIR->print(errs());
            fprintf(stderr, "\n");
            FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
        }
    } else {
        // skip token for error recovery.
        getNextToken();
    }
}

static void handleTopLevelExpression() {
    // Evaluate a top-level expression into an anonymous function.
    if (auto FnAST = parseTopLevelExpr()) {
        if (FnAST->codegen()) {
            // Create a ResourceTracker to track JIT'd memory allocated to our
            // anonymous expression -- That way we can free it after executing.
            auto RT = TheJIT->getMainJITDylib().createResourceTracker();

            auto TSM = ThreadSafeModule(std::move(theModule), std::move(theContext));
            ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
            InitializeModuleAndPassManager();

            // Search the JIT for the __anon_expr symbol.
            auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));

            // Get the symbol's address and cast it to the right type (take no
            // arguments, returns a double) so we can call it as a native function.
            double (*FP)() = (double (*)())(intptr_t)ExprSymbol.getAddress();

            fprintf(stderr, "Evaluated to %f\n", FP());

            // Delete the anonymous expression module from the JIT.
            ExitOnErr(RT->remove());
        }
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
// "Library" functions that can be "extern'd" from user code.
//==--------------------------------------------------------------
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takesa double and returns 0.
extern "C" DLLEXPORT double putchard(double X) {
    fputc((char)X, stderr);
    return 0;
}

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double X) {
    fprintf(stderr, "%f\n", X);
    return 0;
}
//==--------------------------------------------------------------
// Main driver code.
//==--------------------------------------------------------------


int main() {

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    // install standard binary operators.
    // 1 is lowest precedence.
    binopPrecedence['<'] = 10;
    binopPrecedence['+'] = 20;
    binopPrecedence['-'] = 30;
    binopPrecedence['*'] = 40;  // highest.

    // Prime the first token.
    fprintf(stderr, "ready> ");
    getNextToken();

    // creating the JIT.
    TheJIT = ExitOnErr(KaleidoscopeJIT::Create());

    // Make the module, which holds all the code.
    InitializeModuleAndPassManager();

    // Run the main 'Interpreter loop" now.
    mainLoop();

    // Print out all of the generated code.
    theModule->print(errs(), nullptr);

    return 0;
}
