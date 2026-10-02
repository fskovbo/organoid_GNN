"""Batched double-precision tensor operations for curvature preference energies.

Graph shells are prepared once on the CPU. Fractions, conditional reference
means, designs and sparse accommodation solves then stay on the selected device.
The numerical solver changes neither the energy nor its coefficient conventions.
"""
import numpy as np
import torch
from scipy.sparse import block_diag
from src.models.curvature_energy import graph_laplacian


class AccommodationSolve:
    """Jacobi-preconditioned CG with an explicit final residual certificate.

    The disjoint graph systems form one SPD block diagonal matrix. RHS columns
    are independent solves; chunking limits memory without changing the system.
    No dense node-by-node matrices or eigendecompositions are needed.
    """
    def __init__(self, batch, strength, *, rtol=1e-11, max_iterations=512, rhs_chunk_size=64):
        if not 0 < rtol < 1 or max_iterations < 1 or rhs_chunk_size < 1:
            raise ValueError('Invalid accommodation solver settings')
        self.rtol, self.max_iterations, self.rhs_chunk_size = rtol, max_iterations, rhs_chunk_size
        self.diagonal = 1 + batch.degree * strength[batch.graph_id]
        self.matrix = torch.sparse_csr_tensor(batch.crow, batch.col,
            batch.lap_values * strength[batch.edge_graph], size=batch.shape, device=batch.device)
        self.last_iterations = 0
        self.last_relative_residual = 0.

    def apply(self, value):
        return value + torch.sparse.mm(self.matrix, value)

    @torch.no_grad()
    def __call__(self, rhs):
        vector = rhs.ndim == 1
        rhs = rhs[:, None] if vector else rhs
        pieces = []
        for first in range(0, rhs.shape[1], self.rhs_chunk_size):
            b = rhs[:, first:first+self.rhs_chunk_size].contiguous()
            x = torch.zeros_like(b)
            residual = b.clone()
            z = residual / self.diagonal[:, None]
            direction = z.clone()
            rz = (residual*z).sum(0)
            norm2 = b.square().sum(0)
            target = norm2 * self.rtol**2
            tiny = torch.finfo(b.dtype).tiny
            converged = False
            for iteration in range(self.max_iterations):
                ad = self.apply(direction)
                step = rz / (direction*ad).sum(0).clamp_min(tiny)
                x.add_(direction*step)
                residual.sub_(ad*step)
                z = residual / self.diagonal[:, None]
                next_rz = (residual*z).sum(0)
                direction = z + direction*(next_rz/rz.clamp_min(tiny))
                rz = next_rz
                if iteration % 8 == 7 or iteration == self.max_iterations-1:
                    if bool(torch.all(residual.square().sum(0) <= target)):
                        converged = True
                        break
            # Check the true residual, not just the recursively updated one.
            relative = ((b-self.apply(x)).square().sum(0) / norm2.clamp_min(tiny)).sqrt().max()
            error = float(relative)
            self.last_iterations = max(self.last_iterations, iteration+1)
            self.last_relative_residual = max(self.last_relative_residual, error)
            if not converged or not np.isfinite(error) or error > max(5*self.rtol, 5e-14):
                raise RuntimeError(f'Accommodation CG failed: relative residual={error:.3g}, iterations={iteration+1}')
            pieces.append(x)
        result = torch.cat(pieces, dim=1)
        return result[:, 0] if vector else result


class EnergyBatch:
    """Immutable, device-resident graph features shared by fitting and inference."""
    def __init__(self, samples, n_markers, *, device='cuda'):
        if not samples:
            raise ValueError('Empty graph batch')
        self.device = torch.device(device)
        self.n_types = n_markers+1
        self.lengths = np.asarray([len(s['identity']) for s in samples])
        if np.any(self.lengths < 1):
            raise ValueError('Empty graph')
        self.offsets = np.r_[0, self.lengths.cumsum()]
        self.n_graphs = len(samples)
        ids = np.concatenate([s['identity'] for s in samples])
        if not np.issubdtype(ids.dtype,np.integer) or np.any((ids < 0) | (ids >= self.n_types)):
            raise ValueError('Identity outside marker panel')
        counts = np.concatenate([s['counts'][:, :2] for s in samples]).astype(float)
        if counts.shape != (len(ids), 2, self.n_types) or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError('Expected finite nonnegative two-hop identity counts')
        total = counts.sum(1)
        population = total.sum(1, keepdims=True)
        fraction = np.divide(total, population, out=np.zeros_like(total), where=population>0)
        share = np.divide(counts, total[:, None], out=np.zeros_like(counts), where=total[:, None]>0)
        by_graph = np.stack([np.bincount(s['identity'], minlength=self.n_types) for s in samples])
        support = np.maximum((by_graph>0).sum(0), 1)
        ref_weight = np.concatenate([1/(row[s['identity']]*support[s['identity']]) for row,s in zip(by_graph,samples)])
        self.ids = self.tensor(ids, torch.long)
        self.graph_id = self.tensor(np.repeat(np.arange(len(samples)), self.lengths), torch.long)
        self.center = torch.nn.functional.one_hot(self.ids, self.n_types).to(torch.float64)
        self.type_nodes = [self.tensor(np.flatnonzero(ids==a), torch.long) for a in range(self.n_types)]
        self.fraction, self.share = self.tensor(fraction), self.tensor(share)
        ring_population=counts.sum(2,keepdims=True)
        ring_fraction=np.divide(counts,ring_population,out=np.zeros_like(counts),where=ring_population>0)
        self.pooled_fraction=self.tensor((ring_fraction[:,0]+.5*ring_fraction[:,1])/1.5)
        self.pooled_count=self.tensor((counts[:,0]+.5*counts[:,1])/1.5)
        self.first_fraction=self.tensor(ring_fraction[:,0])
        self.first_count=self.tensor(counts[:,0])
        self.reference_weight = self.tensor(ref_weight)
        self.weight = self.tensor(np.repeat(1/(len(samples)*self.lengths), self.lengths))
        self.N = self.tensor([s['N'] for s in samples])
        if not bool(torch.all(torch.isfinite(self.N) & (self.N>0))):
            raise ValueError('N must be positive and finite')
        laps = [s['laplacian'] if 'laplacian' in s else graph_laplacian(s['transition']) for s in samples]
        lap = block_diag(laps, format='csr')
        self.shape = lap.shape
        self.crow, self.col = self.tensor(lap.indptr, torch.long), self.tensor(lap.indices, torch.long)
        self.lap_values, self.degree = self.tensor(lap.data), self.tensor(lap.diagonal())
        self.edge_graph = self.tensor(np.repeat(np.repeat(np.arange(len(samples)),self.lengths),np.diff(lap.indptr)),torch.long)
        self.laplacian = torch.sparse_csr_tensor(self.crow,self.col,self.lap_values,size=lap.shape,device=self.device)

    def tensor(self, value, dtype=torch.float64):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def basis(self, model):
        z = self.N.log()-self.tensor(model.log_reference)
        return torch.stack((torch.ones_like(z),z),dim=1) if model.size_dependent else torch.ones_like(z[:,None])

    def responses(self, model):
        fraction=(self.pooled_count if model.exposure_kind=='counts' else self.pooled_fraction) if model.interaction=='pooled' else self.fraction
        if model.interaction=='pooled' and getattr(model,'interaction_radius',2)==1:
            fraction=self.first_count if model.exposure_kind=='counts' else self.first_fraction
        alpha = self.tensor(model.log_alpha).exp()[self.ids] if model.activation!='linear' else torch.zeros_like(self.fraction)
        safe = alpha.clamp_min(1e-6)
        den, num = safe.tanh(), (safe*fraction).tanh()
        value = torch.where(alpha<1e-5,fraction,num/den)
        gradient = safe*(fraction*(1-num*num)/den-num*(1-den*den)/(den*den))
        gradient = torch.where(alpha<1e-5,0.,gradient)
        if model.activation in ('presence','mixed'):
            mask=torch.ones_like(fraction,dtype=torch.bool) if model.activation=='presence' else self.tensor(np.asarray(model.pair_activations)=='presence',torch.bool)[self.ids]
            value=torch.where(mask,(fraction>0).to(torch.float64),fraction)
            gradient=torch.zeros_like(value)
        if model.interaction=='pooled':
            return torch.stack((value,torch.zeros_like(value)),1),torch.stack((gradient,torch.zeros_like(gradient)),1)
        return value[:,None]*self.share, gradient[:,None]*self.share

    def reference(self, response):
        weighted = response.reshape(len(self.ids),-1)*self.reference_weight[:,None]
        return (self.center.T@weighted).reshape(self.n_types,2,self.n_types)

    def design(self, model, response=None, *, expanded=False):
        values=self._design(model,response,expanded=expanded)
        if getattr(model,'zero_mean_output',False):
            from src.data.target_transforms import center_graph_values
            values=center_graph_values(values,self.graph_id)
        return values

    def _design(self, model, response=None, *, expanded=False):
        if not model.pairs:
            return self.center
        if response is None:
            response = self.responses(model)[0]
        q = response-self.tensor(model.reference)[self.ids]
        basis = self.basis(model)[self.graph_id]
        if expanded:
            parts=[self.center]
            for hop in range(2):
                raw=(self.center[:,:,None]*q[:,hop,None,:]).reshape(len(q),-1)
                parts.append((raw[:,:,None]*basis[:,None,:]).reshape(len(q),-1))
            return torch.cat(parts,dim=1)
        combined = self.combine_response(model,q)
        contrast = self.tensor(model.contrast).reshape(self.n_types,self.n_types,-1)
        projected = torch.empty((len(q),contrast.shape[-1]),dtype=torch.float64,device=self.device)
        for a,nodes in enumerate(self.type_nodes):
            projected.index_copy_(0,nodes,combined.index_select(0,nodes)@contrast[a])
        return torch.cat((self.center,(projected[:,:,None]*basis[:,None,:]).reshape(len(q),-1)),dim=1)

    def combine_response(self,model,response):
        if model.interaction=='pooled':return response[:,0]
        return response[:,0]+.5*self.tensor(model.signs)[self.ids]*response[:,1]

    def solver(self, model, basis=None, **settings):
        if not model.accommodation:
            return None, torch.zeros_like(self.N)
        basis = self.basis(model) if basis is None else basis
        logits = basis@self.tensor(model.lambda_logits)
        strength = (torch.full_like(self.N,model.fixed_strength) if getattr(model,'fixed_strength',None) is not None
                    else torch.logaddexp(torch.zeros_like(logits),logits))
        return AccommodationSolve(self,strength,**settings),strength

    @torch.no_grad()
    def predict(self, model, **solver_settings):
        if not bool(model.is_fitted):
            raise RuntimeError('Model not fitted')
        local = self.design(model)@self.tensor(model.weights)
        solve,_ = self.solver(model,**solver_settings)
        value = (solve(local) if solve is not None else local)*self.tensor(model.target_scale)
        return value


def predict_energy_samples(model, samples, *, device='cuda', **solver_settings):
    """Predict precomputed graph samples; targets are neither read nor required."""
    batch=EnergyBatch(samples,model.n_markers,device=device)
    result=batch.predict(model,**solver_settings).cpu().numpy()
    return [result[a:b] for a,b in zip(batch.offsets[:-1],batch.offsets[1:])]
