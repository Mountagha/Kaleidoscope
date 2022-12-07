#include <string>
#include <memory>
#include <vector>

#include "lexer.hpp"

/// Express AST - Base class for all expressions nodes.
class ExprAST {
    public:
        virtual ~ExprAST() {}
};

/// NumberExprAST - Expression class for numeric literals like "1.0"
class NumberExprAST : public ExprAST {
    double Val;
    public:
        NumberExprAST(double val): Val(val) {}
};

/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
    std::string Name;

    public:
        VariableExprAST(const std::string &name) : Name(name) {}
};

/// BinaryExprAST - Expression class for functions calls.
class BinaryExprAST : public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;
    public:
        BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs, 
                    std::unique_ptr<ExprAST> rhs)
        : Op(op), LHS(std::move(lhs)), RHS(std::move(rhs)) {}

};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;
    public:
        CallExprAST(const std::string &callee, std::vector<std::unique_ptr<ExprAST>> args)
        : Callee(callee), Args(std::move(args)) {}
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
        
        const std::string &getName() const { return Name; }
};

/// FunctionAST - this class represents a function defintion itself.
class FunctionAST {
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprAST> Body;

    public:
        FunctionAST(std::unique_ptr<PrototypeAST> proto, 
                    std::unique_ptr<ExprAST> body)
            : Proto(std::move(Proto)), Body(std::move(Body)) {}
};

/// Parser
static int curTok;
static int getNextToken() {
    return curTok = gettok();
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

    if (curTok = '(') // Simple variable ref.
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
            if (curTok = ')')
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