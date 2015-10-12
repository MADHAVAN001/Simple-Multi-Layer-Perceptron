function out = bipolar_sigmoid_function(a,b,u)
out = a*(-1+2/(1+exp(-b*u)));
end