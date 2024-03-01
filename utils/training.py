""" Author: Dmytro Velychko
    Institution: Carl von Ossietzky University of Oldenburg
    Email: dmytro.velychko@uni-oldenburg.de
"""

from typing import Any
import torch

def train_LBFGS(model, dataloader, 
                nepochs=1000, 
                on_epoch=None, 
                history_size=20, 
                max_iter=20):
    
    lbfgs = torch.optim.LBFGS(model.parameters(), 
        history_size=history_size, 
        max_iter=max_iter, 
        line_search_fn="strong_wolfe")
    
    elbos = []
    gradsizes = []
    for epoch in range(nepochs):
        for batch_idx, (data, indexes) in enumerate(dataloader):
            data = data.type(torch.get_default_dtype()).to(model.device)

            def closure():
                lbfgs.zero_grad()
                elbo = model(data, indexes)
                loss = -elbo
                loss.backward()    
                #print("Allocated CUDA RAM: {}".format(torch.cuda.memory_allocated(0)))
                return loss
            
            loss = lbfgs.step(closure)
            elbos.append(-loss.item())
            
            # Compute gradient size
            grad = torch.hstack([param.grad.flatten() for param in model.parameters() if param.requires_grad])
            gradsize = torch.sqrt((grad**2).sum()).item()
            gradsizes.append(gradsize)
            print('Epoch {:4d} \t Batch {:4d} \t ELBO: {:.6f} \tGradient size: {:.6f}'.format(epoch, batch_idx, -loss.item(), gradsize))
            if len(elbos) > 3 and elbos[-1] == elbos[-3]:
                return elbos, gradsizes
            
        if on_epoch is not None:
            on_epoch(model, epoch+1, elbos)
        
    return elbos, gradsizes


def train_Adam(model, dataloader, lr=1e-3, nepochs=1000,  on_epoch_start=None, on_epoch_finish=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    elbos = []
    gradmeansizes = []
    gradtotalvars = []
    for epoch in range(nepochs):
        
        if on_epoch_start is not None:
            on_epoch_start(model, epoch+1)

        for batch_idx, (data, indexes) in enumerate(dataloader):
            data = data.type(torch.get_default_dtype()).to(model.device)
            optimizer.zero_grad()
            elbo = model(data, indexes)
            elbos.append(elbo.item())
            loss = -elbo
            loss.backward()
            optimizer.step()
            print('Epoch {:4d} \t Batch {:4d} \t ELBO: {:.6f}'.format(epoch, batch_idx, elbo.item()))
    
        if epoch%10 == 0 and False:
            # Estimate total variance of the gradient
            grads = []
            for j in range(5):
                optimizer.zero_grad()
                (-model(data, indexes)).backward()
                grads.append(torch.hstack([param.grad.flatten() for param in model.parameters()]))
            grads = torch.vstack(grads)
            gradmean = grads.mean(0)
            zeromeangrads = grads - gradmean
            gradmeansize = torch.sqrt((gradmean**2).sum()).item()
            gradtotalvar = torch.sqrt((zeromeangrads**2).sum(0) / (zeromeangrads.shape[0] - 1)).sum().item()
            print('Gradient mean: {:.6f} \t variance: {:.6f}'.format(gradmeansize, gradtotalvar))
            gradmeansizes.append(gradmeansize)
            gradtotalvars.append(gradtotalvar)  
        
        if on_epoch_finish is not None:
            on_epoch_finish(model, epoch+1, elbos)

    return elbos, gradmeansizes, gradtotalvars


def optimize_latents(model, data, indexes, init=False, max_iter=20):
    """ E-step. Requires unamortized variational parameters.
    """
    vp = model.variationalparams
    model.variationalparams = vp.__class__(N=data.shape[0], D=vp.D, H=vp.H).to(model.device)
    if init:
        model.variationalparams.set(slice(None), vp, indexes)
    
    if hasattr(model, "on_compute_entropy"):
        oce = model.on_compute_entropy
        model.on_compute_entropy = None
    
    lbfgs = torch.optim.LBFGS(model.variationalparams.parameters(), 
        history_size=20, 
        max_iter=max_iter, 
        line_search_fn="strong_wolfe")
    
    def closure():
        lbfgs.zero_grad()
        elbo = model(data, indexes=None)
        loss = -elbo
        loss.backward()
        return loss
    
    loss = lbfgs.step(closure)

    if hasattr(model, "on_compute_entropy"):
        model.on_compute_entropy = oce
    elbo_item = model(data, indexes=None).item()

    vp.set(indexes, model.variationalparams, slice(None))
    model.variationalparams = vp
    return elbo_item


def adjust_parameters(model, data, indexes, lr=1.0):
    """ M-step. Perform one gradient step.
    """
    model.W.grad.zero_()
    elbo = model(data, indexes)
    elbo.backward()
    with torch.no_grad():
        model.W.add_(lr * model.W.grad)
    return elbo.item()
        

class CallbackList(object):

    def __init__(self, callbacks=None) -> None:
        self.callbacks = []
        if callbacks is not None:
            if isinstance(callbacks, (list, tuple)):
                self.callbacks.extend(callbacks)
            else:
                self.callbacks.append(callbacks)

        self.enabled = True

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.enabled:
            for f in self.callbacks:
                f(*args, **kwds)