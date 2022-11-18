#include <string>

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

