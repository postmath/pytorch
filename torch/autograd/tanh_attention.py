import torch

from torch import Tensor
from torch.autograd.function import Function

from typing import Any

class TanhAttention(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Following the onboarding packet definition. In the absence of broadcasting,
let:   q: B0 x B1 X ... X BP X M x N,
       k: B0 x B1 X ... X BP X K x N,
          then x: B0 x B1 X ... X BP X M x K,
               a: B0 x B1 X ... X BP X M x K,
so let v: B0 x B1 X ... X BP X K x L,
          then o: B0 x B1 X ... X BP X M x L."""
        x = torch.matmul(q, k.transpose(-1, -2))
        a = torch.tanh(x)
        o = torch.matmul(a, v)

        return o, a

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Tensor, Tensor, Tensor], outputs: tuple[Tensor, Tensor]) -> None:
        ctx.save_for_backward(*inputs, *outputs)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx: Any, o_grad: Tensor, a_grad: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """ We have                 
o_grad: B0 x B1 X ... X BP X M x L,
a_grad: B0 x B1 X ... X BP X M x K,
and need v_grad: B0 x B1 X ... X BP X K x L,
         x_grad: B0 x B1 X ... X BP X M x K,
         k_grad: B0 x B1 X ... X BP X K x N,
         q_grad: B0 x B1 X ... X BP X M x N."""
        q, k, v, o, a = ctx.saved_tensors

        if v.requires_grad:
            v_grad = torch.matmul(a.transpose(-1, -2), o_grad)
        else:
            v_grad = None
         
        if q.requires_grad or k.requires_grad:
            # Track the "extra" dependency of a itself on o:
            a_grad += torch.matmul(o_grad, v.transpose(-1, -2))
            
            # The derivative of tanh(x) is 1 - tanh(x)^2, which is convenient because we already
            # know tanh(x) =: a. However, that's numerically unstable for abs(x) large. In that case
            # we rewrite it as 1/cosh(x)^2. We might be able to improve this by making this choice
            # independently per element, but then we have a whole lot of branches...
            if torch.max(torch.abs(a)) > 0.99:
                # We could store x if we use the calling sequence where ctx is an argument of
                # `forward` and `backward`.
                x = torch.matmul(q, k.transpose(-1, -2))
                inv_cosh = 1 / torch.cosh(x)
                tanh_derivatives = inv_cosh * inv_cosh
            else:
                tanh_derivatives = 1 - a * a
            x_grad = a_grad * tanh_derivatives
         
            if q.requires_grad:
                q_grad = torch.matmul(x_grad, k)
            else:
                q_grad = None
         
            if k.requires_grad:
                k_grad = torch.matmul(x_grad.transpose(-1, -2), q)
            else:
                k_grad = None
        else:
            q_grad = None
            k_grad = None

        return q_grad, k_grad, v_grad

# Tests:

# q = torch.randn(1, 1, requires_grad=True, dtype=torch.double)      
# k = torch.randn(1, 1, requires_grad=True, dtype=torch.double)      
# v = torch.randn(1, 1, requires_grad=True, dtype=torch.double)      
#                                                                    
# o, a = torch.autograd.TanhAttention.apply(q, k, v)                                
# loss = o.pow(2).sum() + a.pow(2).sum()
# loss.backward()
#  
# q.grad, k.grad, v.grad
#  
# q = torch.randn(3, 2, requires_grad=True, dtype=torch.double)
# k = torch.randn(5, 2, requires_grad=True, dtype=torch.double)
# v = torch.randn(5, 4, requires_grad=True, dtype=torch.double)
#  
# print(torch.autograd.gradcheck(torch.autograd.TanhAttention.apply, (q, k, v), atol=0.01))
