#include <string>
#include <memory>

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

class CallExprAST : public ExprAST {
    std::string callee;
    std::vector<std::unique_ptr<ExprAST>> Args;
    public:
        CallExprAST(const std::string &callee, std::vector<std::unique_ptr<ExprAST>> args)
        : callee(callee), Args(std::move(args))
};
