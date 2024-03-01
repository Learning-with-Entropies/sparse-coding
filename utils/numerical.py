""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

# See :
# https://en.wikipedia.org/wiki/Error_function#B%C3%BCrmann_series
#
# Roy M. Howard (2022) - Arbitrarily Accurate Analytical Approximations for the Error Function
# https://content.wolfram.com/uploads/sites/19/2014/11/Schoepf.pdf
#
# H. M. Schöpf, P. H. Supancic (2014) - On Bürmann’s Theorem and Its Application to Problems 
# of Linear and Nonlinear Heat Transfer  and Diffusion Expanding a Function in Powers of Its Derivative
# https://www.mdpi.com/2297-8747/27/1/14#B23-mca-27-00014


import math
import torch



def ERFBuermannApproximation2(x):
    expnegxsqr = torch.exp(-x**2)
    return 2 / math.sqrt(math.pi) * torch.sign(x) * torch.sqrt(1-expnegxsqr) * (
            math.sqrt(math.pi) / 2 \
            + 31/200 * expnegxsqr \
            - 341/8000 * expnegxsqr**2
        )
    


def ERFBuermannApproximation4(x):
    oneminusexpnegxsqr = 1 - torch.exp(-x**2)
    return 2 / math.sqrt(math.pi) * torch.sign(x) * torch.sqrt(oneminusexpnegxsqr) * ( 
            1 \
            - 1/12 * oneminusexpnegxsqr \
            - 7/480 * oneminusexpnegxsqr**2 \
            - 5/896 * oneminusexpnegxsqr**3 \
            - 787/276480 * oneminusexpnegxsqr**4
        )


def ERFBuermannApproximation4v2(x):
    oneminusexpnegxsqr = torch.sqrt(1 - torch.exp(-x**2))
    return 2 / math.sqrt(math.pi) * torch.sign(x) * ( 
            oneminusexpnegxsqr \
            - 1/12 * oneminusexpnegxsqr**3 \
            - 7/480 * oneminusexpnegxsqr**5 \
            - 5/896 * oneminusexpnegxsqr**7 \
            - 787/276480 * oneminusexpnegxsqr**3 \
        )


class ErrorFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return ERFBuermannApproximation2(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 / math.sqrt(math.pi) * torch.exp(-input**2)
    

if __name__ == "__main__":
    x = torch.linspace(-5, 5, 1000)
    diff = ERFBuermannApproximation2(x) - torch.erf(x)
    err = torch.max(torch.abs(diff)).item()
    print(err)
    assert err < 0.0037
    