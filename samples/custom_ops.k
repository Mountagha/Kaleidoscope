# Logical unary not.
def unary!(v)
  if v then
    0
  else
    1;

# Unary negate.
def unary-(v)
  0-v;

# Define > with the same precedence as <.
def binary> 10 (LHS RHS)
  RHS < LHS;

# Binary logical or, which does not short circuit.
def binary| 5 (LHS RHS)
  if LHS then
    1
  else if RHS then
    1
  else
    0;

# Binary logical and, which does not short circuit.
def binary& 6 (LHS RHS)
  if !LHS then
    0
  else
    !!RHS;

# Define = with slightly lower precedence than relationals.
def binary = 9 (LHS RHS)
  !(LHS < RHS | LHS > RHS);

# Define ':' for sequencing: as a low-precedence operator that ignores operands
# and just returns the RHS.
def binary : 1 (x y) y;


# the following prints out a character whose "density" reflects the value passed in:
# The lower the value, the denser the character

extern putchard(char);

def printdensity(d)
    if d > 8 then
        putchard(32)    # ' '
    else if d > 4 then
        putchard(46)    # '.'
    else if d > 2 then 
        putchard(43)    # '+'
    else 
        putchard(42);   # '*'
    
printdensity(1): printdensity(2): printdensity(3):
printdensity(4): printdensity(5): printdensity(9):
printdensity(10);