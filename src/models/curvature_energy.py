"""Equal-cell preferred-curvature energy with constrained signed hop responses.

E(h) = .5 sum_i (h_i-u_i)^2 + .5 lambda sum_{unordered edges} (h_i-h_j)^2.
Only graph adjacency is used; there are no patch-area weights or measured-neighbor inputs.
"""
import numpy as np
import torch
from torch import nn
from scipy.linalg import null_space
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
from src.models.coupled_fate import normalized_tanh, neighbor_average_matrix
from src.data.neighborhood_counts import fate_identities, exact_hop_counts


def graph_laplacian(transition):
    adjacency=transition.copy().tocsr();adjacency.data[:]=1.
    adjacency.setdiag(0);adjacency.eliminate_zeros()
    return (diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency).tocsc()


class MeanCurvatureEnergy(nn.Module):
    """Constant center preferences/activation; affine-logN pair amplitudes.

    Hop-1 amplitudes have training-weighted zero row/column means. Hop 2 is
    derived as sign_AB * .5 * hop1, with no independent coefficient table.
    Response centering uses training-only conditional means, equally weighting
    organoids. Accommodation is softplus(intercept+slope*log(N/N_ref)).
    """
    def __init__(self,n_markers,pairs=True,accommodation=True,activation='learned',
                 size_dependent=True,fixed_alpha=3.,min_pair_organoids=20):
        super().__init__()
        if activation not in ('linear','fixed','learned'):raise ValueError('Invalid activation')
        self.n_markers=n_markers;self.pairs=bool(pairs);self.accommodation=bool(accommodation)
        self.activation=activation;self.size_dependent=bool(size_dependent)
        self.fixed_alpha=float(fixed_alpha);self.min_pair_organoids=int(min_pair_organoids)
        self.radius=2 if pairs else 0;self.num_layers=self.radius
        t=n_markers+1;k=2 if size_dependent else 1
        self.n_basis=k;self.hidden_dim=t+(t-1)**2*k*int(pairs)
        values=dict(weights=np.zeros(self.hidden_dim),lambda_logits=np.zeros(k),log_alpha=np.full((t,t),np.log(fixed_alpha)),
            signs=np.ones((t,t)),reference=np.zeros((t,2,t)),contrast=np.zeros((t*t,(t-1)**2)),
            active_pairs=np.zeros((t,t),bool),pair_support=np.zeros((t,t),int),
            log_reference=np.array(0.),log_count_range=np.zeros(2),target_scale=np.array(1.),log_variance=np.array(0.),is_fitted=np.array(False))
        for name,value in values.items():self.register_buffer(name,torch.as_tensor(value))

    @staticmethod
    def array(value):return value.detach().cpu().numpy()

    def basis(self,N):
        n=np.asarray(N,float).reshape(-1)
        if np.any(n<=0) or not np.isfinite(n).all():raise ValueError('N must be positive')
        z=np.log(n)-float(self.log_reference)
        return np.c_[np.ones(len(z)),z] if self.size_dependent else np.ones((len(z),1))

    def strength(self,N):
        return np.logaddexp(0,self.basis(N)@self.array(self.lambda_logits)) if self.accommodation else np.zeros(np.asarray(N).size)

    def alpha(self):return np.exp(self.array(self.log_alpha)) if self.activation!='linear' else np.zeros_like(self.array(self.log_alpha))

    def raw_response(self,sample,derivative=False):
        counts=np.asarray(sample['counts'],float)[:,:2];total=counts.sum(1);population=total.sum(1,keepdims=True)
        fraction=np.divide(total,population,out=np.zeros_like(total),where=population>0)
        share=np.divide(counts,total[:,None,:],out=np.zeros_like(counts),where=total[:,None,:]>0)
        value,gradient=normalized_tanh(fraction,self.alpha()[sample['identity']],True)
        return (value[:,None,:]*share,gradient[:,None,:]*share) if derivative else value[:,None,:]*share

    def refresh_reference(self,samples):
        t=self.n_markers+1;mean=np.zeros((t,2,t));derivative=mean.copy();den=np.zeros(t)
        for s in samples:
            q,dq=self.raw_response(s,True)
            for a in np.unique(s['identity']):
                take=s['identity']==a;mean[a]+=q[take].mean(0);derivative[a]+=dq[take].mean(0);den[a]+=1
        mean/=np.maximum(den[:,None,None],1);derivative/=np.maximum(den[:,None,None],1)
        self.reference.copy_(torch.as_tensor(mean));return derivative

    def configure(self,samples):
        if not samples:raise ValueError('Empty fitting set')
        t=self.n_markers+1;logs=np.log([s['N'] for s in samples])
        self.log_reference.fill_(logs.mean());self.log_count_range.copy_(torch.tensor([logs.min(),logs.max()]))
        self.target_scale.fill_(max(np.sqrt(np.mean([np.mean(s['y']**2) for s in samples])),1e-8))
        center=np.maximum(np.mean([np.bincount(s['identity'],minlength=t)/len(s['identity']) for s in samples],axis=0),1e-8);center/=center.sum()
        source=np.zeros(t);support=np.zeros((t,t),int)
        for s in samples:
            counts=s['counts'][:,:2].sum(1);n=counts.sum(1,keepdims=True)
            source+=np.divide(counts,n,out=np.zeros_like(counts,dtype=float),where=n>0).mean(0)
            for a in np.unique(s['identity']):support[a]+=np.any(counts[s['identity']==a]>0,axis=0)
        source=np.maximum(source,1e-8);source/=source.sum()
        q=np.einsum('ad,be->abde',null_space(center[None]),null_space(source[None])).reshape(t*t,(t-1)**2)
        self.contrast.copy_(torch.tensor(q));self.pair_support.copy_(torch.tensor(support))
        self.active_pairs.copy_(torch.tensor((support>=self.min_pair_organoids)&self.pairs))
        self.refresh_reference(samples);return self

    def raw_design(self,sample):
        ids=sample['identity'];t=self.n_markers+1;center=np.eye(t)[ids]
        if not self.pairs:return center
        q=self.raw_response(sample)-self.array(self.reference)[ids]
        basis=self.basis([sample['N']])[0]
        pieces=[center]
        for r in range(2):
            v=np.zeros((len(ids),t,t));v[np.arange(len(ids)),ids,:]=q[:,r]
            pieces.append((v.reshape(len(ids),t*t,1)*basis).reshape(len(ids),-1))
        return np.concatenate(pieces,axis=1)

    def mapping(self,signs=None):
        t=self.n_markers+1
        if not self.pairs:return np.eye(t)
        block=np.kron(self.array(self.contrast),np.eye(self.n_basis));size=len(block)
        transform=np.zeros((t+2*size,self.hidden_dim));transform[:t,:t]=np.eye(t);transform[t:t+size,t:]=block
        sign=self.array(self.signs) if signs is None else signs
        transform[t+size:,t:]=.5*np.repeat(sign.ravel(),self.n_basis)[:,None]*block
        return transform

    def design(self,sample):
        ids=sample['identity'];t=self.n_markers+1;center=np.eye(t)[ids]
        if not self.pairs:return center
        q=self.raw_response(sample)-self.array(self.reference)[ids]
        combined=q[:,0]+.5*self.array(self.signs)[ids]*q[:,1]
        contrast=self.array(self.contrast).reshape(t,t,-1)
        projected=np.einsum('nb,nbp->np',combined,contrast[ids])
        pairs=(projected[:,:,None]*self.basis([sample['N']])[0]).reshape(len(ids),-1)
        return np.c_[center,pairs]

    def coefficients(self,N):
        b=self.basis(N);t=self.n_markers+1;scale=float(self.target_scale);w=self.array(self.weights)
        a=np.broadcast_to(w[:t]*scale,(len(b),t)).copy()
        pair=np.zeros((len(b),t,t))
        if self.pairs:pair=(b@w[t:].reshape(-1,self.n_basis).T@self.array(self.contrast).T).reshape(len(b),t,t)*scale
        return dict(center=a,hop1=pair,hop2=.5*self.array(self.signs)[None]*pair,alpha=self.alpha(),accommodation=self.strength(N))

    def predict_samples(self,samples,*,device=None,**solver_settings):
        """Batched inference; CUDA reuses the certified tensor energy solver."""
        device=self.weights.device if device is None else torch.device(device)
        if device.type=='cpu':
            return [self._predict_sample_cpu(sample) for sample in samples]
        from src.models.energy_ops import predict_energy_samples
        return predict_energy_samples(self,samples,device=device,**solver_settings)

    def predict_sample(self,sample,N=None,*,device=None,**solver_settings):
        sample=sample if N is None else dict(sample,N=N)
        return self.predict_samples([sample],device=device,**solver_settings)[0]

    def _predict_sample_cpu(self,sample,N=None):
        if not bool(self.is_fitted):raise RuntimeError('Model not fitted')
        s=sample if N is None else dict(sample,N=N)
        local=self.design(s)@self.array(self.weights)
        lam=float(self.strength([s['N']])[0]);lap=s.get('laplacian')
        if lap is None:lap=graph_laplacian(s['transition'])
        prediction=splu(eye(len(local),format='csc')+lam*lap).solve(local) if lam else local
        return prediction*float(self.target_scale)

    def forward(self,x,edge_index,data=None):
        ids=fate_identities(x);edges=self.array(edge_index).astype(int);batch=getattr(data,'batch',None)
        batch=np.zeros(len(x),int) if batch is None else self.array(batch).astype(int)
        supplied=getattr(data,'full_num_cells',None)
        sizes=None if supplied is None else np.asarray(self.array(supplied) if torch.is_tensor(supplied) else supplied).reshape(-1)
        if edges.size and np.any(batch[edges[0]]!=batch[edges[1]]):raise ValueError('Cross-graph edges')
        result=np.zeros(len(x));samples=[];node_groups=[]
        for index in np.unique(batch):
            nodes=np.flatnonzero(batch==index);mapping=np.full(len(x),-1);mapping[nodes]=np.arange(len(nodes))
            edge=mapping[edges[:,batch[edges[0]]==index]]
            s=dict(N=len(nodes) if sizes is None else float(sizes[index]),identity=ids[nodes],counts=exact_hop_counts(x[nodes],edge,2),transition=neighbor_average_matrix(len(nodes),edge))
            samples.append(s);node_groups.append(nodes)
        for nodes,prediction in zip(node_groups,self.predict_samples(samples,device=x.device)):
            result[nodes]=prediction
        mu=torch.as_tensor(result,device=x.device);return (mu,self.log_variance.to(x.device).expand_as(mu)),mu[:,None]
