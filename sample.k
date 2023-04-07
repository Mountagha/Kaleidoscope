# compute the x'th fibonnaci number
def fib(x)
    if x < 3 then
        1
    else 
        fib(x-1) + fib(x-2);

# this expression will compute the 40th number
fib(40);