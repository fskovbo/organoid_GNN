"""Equal-cell preferred-curvature energy with signed-hop or pooled fate responses.

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
    Alternatively, pooled interactions activate fixed weighted ring exposures
    once, with a single amplitude table and optional reference subtraction.
    Direct pair amplitudes remove weighted-zero constraints and can select a
    subset of source and recipient identities; excluded pairs are exactly zero.
    Fixed-strength fits store accommodation directly, including exact zero.
    The default retains the historical contrast basis and checkpoint shapes.
    Response centering uses training-only conditional means, equally weighting
    organoids. Accommodation is softplus(intercept+slope*log(N/N_ref)).
    """
    def __init__(self,n_markers,pairs=True,accommodation=True,activation='learned',
                 size_dependent=True,fixed_alpha=3.,min_pair_organoids=20,
                 interaction='signed_hops',center_response=True,exposure_kind='fractions',
                 pair_constraints='weighted_contrasts',source_indices=None,
                 interaction_radius=2,reference_source=None,pair_activations=None,
                 recipient_indices=None,fixed_strength=None,zero_mean_output=False):
        super().__init__()
        if activation not in ('linear','fixed','learned','presence','mixed'):raise ValueError('Invalid activation')
        if interaction_radius not in (1,2):raise ValueError('Interaction radius must be 1 or 2')
        if (interaction_radius!=2 or activation in ('presence','mixed')) and interaction!='pooled':
            raise ValueError('Radius selection and presence responses require pooled interactions')
        if interaction not in ('signed_hops','pooled'):raise ValueError('Invalid interaction construction')
        if exposure_kind not in ('fractions','counts'):raise ValueError('Invalid exposure kind')
        if exposure_kind=='counts' and interaction!='pooled':raise ValueError('Count exposure requires pooled interactions')
        if pair_constraints not in ('weighted_contrasts','direct'):raise ValueError('Invalid pair constraints')
        if source_indices is not None and pair_constraints!='direct':raise ValueError('Source selection requires direct pair amplitudes')
        sources=list(range(n_markers+1)) if source_indices is None else list(source_indices)
        if not sources or len(set(sources))!=len(sources) or any(not isinstance(b,(int,np.integer)) or b<0 or b>n_markers for b in sources):
            raise ValueError('Expected distinct source identity indices within marker panel')
        if reference_source is not None:
            if pair_constraints!='direct' or activation!='linear' or reference_source not in sources:
                raise ValueError('A source reference requires direct linear interactions and an included source')
            if len(sources)!=n_markers+1:raise ValueError('A source reference requires all identities')
        self.pair_activations=None
        if activation=='mixed':
            kinds=np.asarray(pair_activations)
            if kinds.shape!=(n_markers+1,n_markers+1) or not np.isin(kinds,['linear','presence']).all():
                raise ValueError('Mixed activation needs one linear/presence choice per ordered pair')
            self.pair_activations=kinds.tolist()
        elif pair_activations is not None:raise ValueError('Pair activation map requires activation=mixed')
        if recipient_indices is not None and pair_constraints!='direct':
            raise ValueError('Recipient selection requires direct pair amplitudes')
        recipients=list(range(n_markers+1)) if recipient_indices is None else list(recipient_indices)
        if not recipients or len(set(recipients))!=len(recipients) or any(not isinstance(a,(int,np.integer)) or a<0 or a>n_markers for a in recipients):
            raise ValueError('Expected distinct recipient identity indices within marker panel')
        if fixed_strength is not None and (not np.isfinite(fixed_strength) or fixed_strength<0 or not accommodation or size_dependent):
            raise ValueError('Fixed accommodation requires a finite nonnegative scalar, accommodation=True and size_dependent=False')
        if zero_mean_output and (fixed_strength is None or interaction!='pooled' or activation not in ('linear','presence','fixed','mixed')):
            raise ValueError('Zero-mean output currently requires fixed accommodation and fixed pooled activation')
        self.zero_mean_output=bool(zero_mean_output)
        self.recipient_indices=None if recipient_indices is None else [int(a) for a in recipients]
        self.fixed_strength=None if fixed_strength is None else float(fixed_strength)
        self.reference_source=reference_source
        self.interaction_radius=int(interaction_radius)
        self.source_indices=None if source_indices is None else [int(b) for b in sources]
        self.pair_constraints=pair_constraints
        self.exposure_kind=exposure_kind
        self.interaction=interaction;self.center_response=bool(center_response)
        self.n_markers=n_markers;self.pairs=bool(pairs);self.accommodation=bool(accommodation)
        self.activation=activation;self.size_dependent=bool(size_dependent)
        self.fixed_alpha=float(fixed_alpha);self.min_pair_organoids=int(min_pair_organoids)
        self.radius=interaction_radius if pairs else 0;self.num_layers=self.radius
        t=n_markers+1;k=2 if size_dependent else 1
        n_pair=(t-1)**2 if pair_constraints=='weighted_contrasts' else len(recipients)*(len(sources)-int(reference_source is not None))
        self.n_basis=k;self.hidden_dim=t+n_pair*k*int(pairs)
        values=dict(weights=np.zeros(self.hidden_dim),lambda_logits=np.zeros(k),log_alpha=np.full((t,t),np.log(fixed_alpha)),
            signs=np.ones((t,t)),reference=np.zeros((t,2,t)),contrast=np.zeros((t*t,n_pair)),
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
        if getattr(self,'fixed_strength',None) is not None:return np.full(np.asarray(N).size,self.fixed_strength)
        return np.logaddexp(0,self.basis(N)@self.array(self.lambda_logits)) if self.accommodation else np.zeros(np.asarray(N).size)

    def alpha(self):return np.exp(self.array(self.log_alpha)) if self.activation!='linear' else np.zeros_like(self.array(self.log_alpha))

    def exposure(self,sample):
        """First-ring exposure, or fixed (1, 1/2) weighted rings divided by 1.5."""
        counts=np.asarray(sample['counts'],float)[:,:2]
        if getattr(self,'interaction_radius',2)==1:
            counts=counts[:,0]
            population=counts.sum(1,keepdims=True)
            return counts if self.exposure_kind=='counts' else np.divide(counts,population,out=np.zeros_like(counts),where=population>0)
        if self.exposure_kind=='counts':return (counts[:,0]+.5*counts[:,1])/1.5
        population=counts.sum(2,keepdims=True)
        fractions=np.divide(counts,population,out=np.zeros_like(counts),where=population>0)
        return (fractions[:,0]+.5*fractions[:,1])/1.5

    def combine_response(self,response,ids):
        if self.interaction=='pooled':return response[:,0]
        return response[:,0]+.5*self.array(self.signs)[ids]*response[:,1]

    def raw_response(self,sample,derivative=False):
        if self.interaction=='pooled':
            exposure=self.exposure(sample)
            value,gradient=normalized_tanh(exposure,self.alpha()[sample['identity']],True,
                input_max=None if self.exposure_kind=='counts' else 1.)
            if self.activation in ('presence','mixed'):
                mask=True if self.activation=='presence' else np.asarray(self.pair_activations)[sample['identity']]=='presence'
                value=np.where(mask,exposure>0,exposure).astype(float)
                gradient=np.zeros_like(value)
            # Keep the legacy buffer layout; pooled activation occupies slot zero.
            value=np.stack((value,np.zeros_like(value)),axis=1)
            gradient=np.stack((gradient,np.zeros_like(gradient)),axis=1)
            return (value,gradient) if derivative else value
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
        if not self.center_response:mean.fill(0.);derivative.fill(0.)
        self.reference.copy_(torch.as_tensor(mean));return derivative

    def configure(self,samples):
        if not samples:raise ValueError('Empty fitting set')
        t=self.n_markers+1;logs=np.log([s['N'] for s in samples])
        self.log_reference.fill_(logs.mean());self.log_count_range.copy_(torch.tensor([logs.min(),logs.max()]))
        self.target_scale.fill_(max(np.sqrt(np.mean([np.mean(s['y']**2) for s in samples])),1e-8))
        center=np.maximum(np.mean([np.bincount(s['identity'],minlength=t)/len(s['identity']) for s in samples],axis=0),1e-8);center/=center.sum()
        source=np.zeros(t);support=np.zeros((t,t),int)
        for s in samples:
            counts=s['counts'][:,:getattr(self,'interaction_radius',2)].sum(1);n=counts.sum(1,keepdims=True)
            source+=np.divide(counts,n,out=np.zeros_like(counts,dtype=float),where=n>0).mean(0)
            for a in np.unique(s['identity']):support[a]+=np.any(counts[s['identity']==a]>0,axis=0)
        source=np.maximum(source,1e-8);source/=source.sum()
        q=np.einsum('ad,be->abde',null_space(center[None]),null_space(source[None])).reshape(t*t,(t-1)**2)
        included=np.ones((t,t),bool)
        if self.pair_constraints=='direct':
            sources=list(range(t)) if self.source_indices is None else self.source_indices
            recipients=list(range(t)) if getattr(self,'recipient_indices',None) is None else self.recipient_indices
            columns=[a*t+b for a in recipients for b in sources if b!=getattr(self,'reference_source',None)]
            q=np.eye(t*t)[:,columns]
            included[:]=False;included[np.ix_(recipients,sources)]=True
            if getattr(self,'reference_source',None) is not None:included[:,self.reference_source]=False
        self.contrast.copy_(torch.tensor(q));self.pair_support.copy_(torch.tensor(support))
        self.active_pairs.copy_(torch.tensor((support>=self.min_pair_organoids)&self.pairs&included))
        self.refresh_reference(samples);return self

    def raw_design(self,sample):
        values=self._raw_design(sample)
        return values-values.mean(axis=0,keepdims=True) if getattr(self,'zero_mean_output',False) else values

    def _raw_design(self,sample):
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
        transform[t+size:,t:]=.5*np.repeat(sign.ravel(),self.n_basis)[:,None]*block if self.interaction=='signed_hops' else 0.
        return transform

    def design(self,sample):
        """Optionally project out the graph mean for conditional allocation tasks.

        Symmetric graph accommodation preserves constants, so projecting before
        propagation is equivalent to centering the predicted field afterward.
        No observed target or curvature summary enters this projection.
        """
        values=self._design(sample)
        return values-values.mean(axis=0,keepdims=True) if getattr(self,'zero_mean_output',False) else values

    def _design(self,sample):
        ids=sample['identity'];t=self.n_markers+1;center=np.eye(t)[ids]
        if not self.pairs:return center
        q=self.raw_response(sample)-self.array(self.reference)[ids]
        combined=self.combine_response(q,ids)
        contrast=self.array(self.contrast).reshape(t,t,-1)
        projected=np.einsum('nb,nbp->np',combined,contrast[ids])
        pairs=(projected[:,:,None]*self.basis([sample['N']])[0]).reshape(len(ids),-1)
        return np.c_[center,pairs]

    def coefficients(self,N):
        b=self.basis(N);t=self.n_markers+1;scale=float(self.target_scale);w=self.array(self.weights)
        a=np.broadcast_to(w[:t]*scale,(len(b),t)).copy()
        pair=np.zeros((len(b),t,t))
        if self.pairs:pair=(b@w[t:].reshape(-1,self.n_basis).T@self.array(self.contrast).T).reshape(len(b),t,t)*scale
        if self.interaction=='pooled':
            return dict(center=a,amplitude=pair,alpha=self.alpha(),accommodation=self.strength(N))
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
