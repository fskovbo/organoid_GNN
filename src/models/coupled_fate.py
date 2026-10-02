"""Explicit fate preferences with predicted or explicitly measured neighbor curvature.

For one organoid: K = (1-gamma) local + gamma W K. W is the one-hop
row-normalized undirected adjacency. Isolated cells use W_ii=1, hence K=local.
Measured mode is a separate conditional diagnostic with observed neighboring targets.
This is a spatial coherence model, not a discretization of Helfrich mechanics.
"""
import copy
import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.linalg import null_space
from scipy.special import expit
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import splu
from torch import nn
from src.data.neighborhood_counts import fate_identities, exact_hop_counts


def neighbor_average_matrix(n_nodes, edge_index):
    """Unique undirected adjacency, no self edges; isolates preserve local value."""
    edges = edge_index.detach().cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype,np.integer):
        raise ValueError('Integer edge_index with shape (2, edges) required.')
    if n_nodes < 1 or (edges.size and (edges.min()<0 or edges.max()>=n_nodes)):
        raise ValueError('Invalid node count or edge index.')
    a,b = edges[:,edges[0]!=edges[1]]
    adjacency = coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(n_nodes,n_nodes)).tocsr()
    adjacency.data[:] = 1.
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    adjacency += diags((degree==0).astype(float))
    degree = np.maximum(degree,1.)
    return (diags(1/degree)@adjacency).tocsr()


def pair_response(fraction, specification):
    """Serializable response family; linear by default, optional pair overrides.

    Presence detects any source-positive cell, not ring population size.
    Hill/threshold responses are normalized to zero at p=0 and one at p=1.
    """
    p = np.asarray(fraction,dtype=float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1+1e-12).any():
        raise ValueError('Fractions must lie in [0,1].')
    kind = specification.get('response','linear')
    if kind=='linear': return p
    if kind=='presence': return (p>0).astype(float)
    if kind=='hill':
        half = float(specification.get('half',.1))
        if not np.isfinite(half) or half<=0: raise ValueError('Hill half must be positive.')
        return (1+half)*p/(p+half)
    if kind=='threshold':
        threshold,width = float(specification.get('threshold',.1)),float(specification.get('width',.02))
        if not 0<threshold<1 or not np.isfinite(width) or width<=0: raise ValueError('Invalid threshold/width.')
        lo,hi = expit(-threshold/width),expit((1-threshold)/width)
        if hi-lo<1e-12: raise ValueError('Threshold width is too large to resolve.')
        return (expit((p-threshold)/width)-lo)/(hi-lo)
    raise ValueError(f'Unknown pair response: {kind}')


class CoupledFateCurvature(nn.Module):
    """Size-dependent center preferences and optional constrained pair tables.

    radius=0 means center preferences only; positive radius adds all exact rings
    up to radius. Coupling makes the effective reach global on each component.
    Pair table rows/columns have training-weighted zero means. With heterogeneous
    response functions these constrain coefficients, not averaged function values;
    inspect complete response curves rather than comparing unlike coefficients.
    """
    def __init__(self,n_markers,radius=0,n_splines=5,coupling=True,gamma_max=.95,pair_responses=None,neighbor_curvature="predicted"):
        super().__init__()
        if not isinstance(n_markers,int) or n_markers<1 or not isinstance(radius,int) or radius<0:
            raise ValueError('Invalid marker count or pair radius.')
        if n_splines<4 or not 0<gamma_max<1:
            raise ValueError('Need >=4 cubic basis functions and 0<gamma_max<1.')
        if neighbor_curvature not in ('predicted','measured'):
            raise ValueError('neighbor_curvature must be predicted or measured.')
        if neighbor_curvature=='measured' and not coupling:
            raise ValueError('Measured-neighbor models require coupling=True.')
        self.neighbor_curvature=neighbor_curvature
        self.n_markers,self.radius,self.n_splines = n_markers,radius,n_splines
        self.coupling,self.gamma_max = bool(coupling),float(gamma_max)
        self.pair_responses = copy.deepcopy(pair_responses or [])
        seen=set()
        for spec in self.pair_responses:
            key=(spec['center'],spec['source'],spec['hop'])
            if key in seen or not all(isinstance(v,int) for v in key) or not (0<=key[0]<=n_markers and 0<=key[1]<=n_markers and 1<=key[2]<=radius):
                raise ValueError('Pair overrides require unique valid center/source/hop indices.')
            if set(spec)-{'center','source','hop','response','half','threshold','width'}:
                raise ValueError('Unknown pair-response setting.')
            pair_response([0.,.5,1.],spec)
            seen.add(key)
        t,d = n_markers+1,n_markers
        self.num_layers=radius  # Explicit local pair radius, NOT effective propagation reach.
        self.hidden_dim=t+radius*d*d
        for name,value in dict(weights=np.zeros((self.hidden_dim,n_splines)),gamma_logits=np.zeros(n_splines),
            knots=np.zeros(n_splines+4),log_count_range=np.zeros(2),center_reference=np.ones(t)/t,
            source_reference=np.ones((radius,t))/t,center_contrast=np.zeros((t,d)),
            source_contrast=np.zeros((radius,t,d)),response_reference=np.zeros((t,radius,t))).items():
            self.register_buffer(name,torch.tensor(value,dtype=torch.float64))
        self.register_buffer('log_variance',torch.tensor(0.,dtype=torch.float64))
        self.register_buffer('is_fitted',torch.tensor(False))

    @staticmethod
    def _array(value): return value.detach().cpu().numpy()

    def basis(self,N):
        n=np.asarray(N,dtype=float).reshape(-1)
        if not np.isfinite(n).all() or (n<=0).any(): raise ValueError('Positive finite N required.')
        lo,hi=self._array(self.log_count_range)
        if hi<=lo: raise RuntimeError('Configure the training size basis first.')
        return BSpline.design_matrix(np.clip(np.log(n),lo,hi),self._array(self.knots),3).toarray()

    def gamma(self,N):
        return self.gamma_max*expit(self.basis(N)@self._array(self.gamma_logits)) if self.coupling else np.zeros(np.asarray(N).size)

    def fractions(self,counts):
        x=np.asarray(counts,dtype=float)[:,:self.radius]
        if x.shape[1:]!=(self.radius,self.n_markers+1) or not np.isfinite(x).all() or (x<0).any():
            raise ValueError('Invalid exact-hop counts.')
        total=x.sum(axis=-1,keepdims=True)
        return np.divide(x,total,out=np.zeros_like(x),where=total>0),total[...,0]>0

    def response_features(self,identities,counts,centered=True):
        fractions,nonempty=self.fractions(counts)
        result=fractions.copy()
        for spec in self.pair_responses:
            take=np.asarray(identities)==spec['center'];r,b=spec['hop']-1,spec['source']
            result[take,r,b]=pair_response(fractions[take,r,b],spec)
        if centered: result-=self._array(self.response_reference)[identities]
        return result*nonempty[...,None]

    def configure(self,samples):
        if not samples: raise ValueError('No training organoids.')
        t=self.n_markers+1
        z=np.log([s['N'] for s in samples]);lo,hi=z.min(),z.max()
        if hi<=lo: raise ValueError('Need at least two distinct training sizes.')
        knots=np.r_[np.repeat(lo,4),np.linspace(lo,hi,self.n_splines-2)[1:-1],np.repeat(hi,4)]
        center=np.maximum(np.mean([np.bincount(s['identity'],minlength=t)/len(s['identity']) for s in samples],axis=0),1e-8)
        center/=center.sum()
        source=np.zeros((self.radius,t));ref=np.zeros((t,self.radius,t));den=np.zeros((t,self.radius,1));sd=np.zeros(self.radius)
        for s in samples:
            frac,nonempty=self.fractions(s['counts']);raw=self.response_features(s['identity'],s['counts'],False)
            for r in range(self.radius):
                if nonempty[:,r].any(): source[r]+=frac[nonempty[:,r],r].mean(axis=0);sd[r]+=1
                for a in range(t):
                    take=(s['identity']==a)&nonempty[:,r]
                    if take.any(): ref[a,r]+=raw[take,r].mean(axis=0);den[a,r]+=1
        source=np.maximum(source/np.maximum(sd[:,None],1),1e-8);source/=source.sum(axis=1,keepdims=True)
        ref=np.divide(ref,den,out=np.zeros_like(ref),where=den>0)
        values=dict(knots=knots,log_count_range=[lo,hi],center_reference=center,source_reference=source,
                    center_contrast=null_space(center[None,:]),source_contrast=np.asarray([null_space(v[None,:]) for v in source]).reshape(self.radius,t,t-1),response_reference=ref)
        for k,v in values.items():getattr(self,k).copy_(torch.as_tensor(v,dtype=torch.float64))
        self.is_fitted.fill_(False)
        return self

    def local_design(self,identities,counts):
        ids=np.asarray(identities,dtype=int)
        center=np.eye(self.n_markers+1)[ids]
        q=self.response_features(ids,counts)
        projected=np.einsum('nrb,rbd->nrd',q,self._array(self.source_contrast))
        pairs=np.einsum('na,nrb->nrab',self._array(self.center_contrast)[ids],projected).reshape(len(ids),-1)
        return np.c_[center,pairs]

    def propagate(self,local,transition,N):
        """Solve for one or many right-hand sides without ever reading targets."""
        gamma=float(self.gamma([N])[0])
        if gamma==0:return np.asarray(local).copy()
        solver=splu((eye(transition.shape[0],format='csc')-gamma*transition).tocsc())
        return solver.solve((1-gamma)*np.asarray(local))

    def predict_sample(self,sample,N=None):
        if not self.is_fitted.item():raise RuntimeError('Model is not fitted.')
        n=sample['N'] if N is None else N
        local=self.local_design(sample['identity'],sample['counts'])@self._array(self.weights)@self.basis([n])[0]
        if self.neighbor_curvature=='measured':
            measured,available=self.measured_neighbors(sample)
            g=float(self.gamma([n])[0])*available
            return (1-g)*local+g*measured
        return self.propagate(local,sample['transition'],n)

    @staticmethod
    def measured_neighbors(sample):
        """Observed-neighbor diagnostic input; never use the center target itself.

        Isolated nodes fall back to the local fate predictor. The self-loop
        convention used by the equilibrium solver must not leak their targets.
        """
        y=np.asarray(sample['y'],dtype=float).reshape(-1)
        w=sample['transition']
        if len(y)!=w.shape[0] or not np.isfinite(y).all():
            raise ValueError('Measured-neighbor inference requires finite observed curvatures.')
        available=np.asarray(w.diagonal()==0,dtype=float)
        if np.any((w.diagonal()!=0)&(w.diagonal()!=1)):
            raise ValueError('Unexpected diagonal in the neighbor matrix.')
        return (w@y)*available,available

    def coefficients(self,N):
        if not self.is_fitted.item():raise RuntimeError('Model is not fitted.')
        w=self.basis(N)@self._array(self.weights).T;t=self.n_markers+1;d=t-1
        pairs=np.einsum('ad,nrde,rbe->nrab',self._array(self.center_contrast),w[:,t:].reshape(len(w),self.radius,d,d),self._array(self.source_contrast))
        return dict(a=w[:,:t],P=pairs,gamma=self.gamma(N))

    def contributions(self,sample,N=None):
        """Local and propagated components; both decompositions sum exactly."""
        n=sample['N'] if N is None else N;co=self.coefficients([n]);ids=sample['identity']
        q=self.response_features(ids,sample['counts'])
        center=co['a'][0,ids];pairs=q*co['P'][0].transpose(1,0,2)[ids]
        columns=np.c_[center,pairs.reshape(len(ids),-1)]
        gamma=co['gamma'][0]
        if self.neighbor_curvature=='measured':
            measured,available=self.measured_neighbors(sample)
            spread=(1-gamma*available[:,None])*columns
            coupling_term=gamma*available*measured
            prediction=spread.sum(axis=1)+coupling_term
        else:
            spread=self.propagate(columns,sample['transition'],n)
            prediction=spread.sum(axis=1)
            coupling_term=gamma*(sample['transition']@prediction)
        return dict(local_center=center,local_pairs=pairs,local=columns.sum(axis=1),
                    propagated_center=spread[:,0],propagated_pairs=spread[:,1:].reshape(pairs.shape),
                    neighbor_prediction=sample['transition']@prediction,coupling_term=coupling_term,
                    prediction=prediction)

    def forward(self,x,edge_index,data=None):
        if not self.is_fitted.item():raise RuntimeError('Model is not fitted.')
        ids=fate_identities(x)
        batch=getattr(data,'batch',None) if data is not None else None
        batch=self._array(batch).astype(int) if batch is not None else np.zeros(len(x),dtype=int)
        edge=self._array(edge_index).astype(int)
        if edge.size and np.any(batch[edge[0]]!=batch[edge[1]]):raise ValueError('Cross-organoid edges in batch.')
        supplied=getattr(data,'full_num_cells',None) if data is not None else None
        sizes=np.bincount(batch) if supplied is None else np.asarray(self._array(supplied) if torch.is_tensor(supplied) else supplied).reshape(-1)
        if len(sizes)!=batch.max()+1:raise ValueError('One full_num_cells value per graph required.')
        predictions=np.zeros(len(x));locals_=np.zeros(len(x))
        for k,n in enumerate(sizes):
            nodes=np.flatnonzero(batch==k);mapping=np.full(len(x),-1);mapping[nodes]=np.arange(len(nodes))
            subedge=mapping[edge[:,batch[edge[0]]==k]]
            sample=dict(N=n,identity=ids[nodes],counts=exact_hop_counts(x[nodes],subedge,self.radius),transition=neighbor_average_matrix(len(nodes),subedge))
            if self.neighbor_curvature=='measured':
                target=getattr(data,'y',None)
                if target is None:raise ValueError('Measured-neighbor model requires data.y in its saved target units.')
                sample['y']=self._array(target).reshape(-1)[nodes]
            predictions[nodes]=self.predict_sample(sample)
            locals_[nodes]=self.local_design(sample['identity'],sample['counts'])@self._array(self.weights)@self.basis([n])[0]
        mu=torch.as_tensor(predictions,dtype=torch.float64,device=x.device)
        return (mu,self.log_variance.to(x.device).expand_as(mu)),torch.as_tensor(locals_[:,None],device=x.device)


def normalized_tanh(x, alpha, derivative=False, *, input_max=1.):
    """Unit-input normalized activation; derivative is with respect to log(alpha).

    By default input is a fraction. Set input_max=None for nonnegative counts;
    F(1)=1 still holds, but F(x) can exceed one when x>1.
    """
    x=np.asarray(x,dtype=float);alpha=np.asarray(alpha,dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(alpha).all() or np.any(x<0) or np.any(alpha<0):raise ValueError('Invalid exposure or saturation.')
    if input_max is not None and np.any(x>input_max+1e-12):raise ValueError('Exposure exceeds its allowed range.')
    safe=np.maximum(alpha,1e-6);den=np.tanh(safe);num=np.tanh(safe*x)
    value=np.where(alpha<1e-5,x,num/den)
    if not derivative:return value
    grad=safe*(x*(1-num*num)/den-num*(1-den*den)/(den*den))
    return value,np.where(alpha<1e-5,0.,grad)


class CoupledFateResponse(CoupledFateCurvature):
    """Whole-neighborhood activation with target-specific spatial propagation.

    Targets share saturation only. Coefficients are stored in training-scaled
    residual units and exposed in physical units. The legacy class is unchanged.
    """
    def __init__(self,n_markers,radius=2,n_splines=5,coupling=True,gamma_max=.95,
                 activation='linear',fixed_alpha=3.,size_dependent=True,n_targets=1,
                 min_pair_organoids=20):
        super().__init__(n_markers,radius,max(4,n_splines),coupling,gamma_max)
        if activation not in ('linear','fixed','learned'):raise ValueError('Unknown activation.')
        if not np.isfinite(fixed_alpha) or fixed_alpha<=0 or n_targets<1 or min_pair_organoids<1:
            raise ValueError('Positive fixed alpha, target count and pair support required.')
        if size_dependent and n_splines<4:raise ValueError('Cubic size dependence needs at least four basis functions.')
        self.activation,self.fixed_alpha=activation,float(fixed_alpha)
        self.size_dependent,self.n_targets=bool(size_dependent),int(n_targets)
        self.min_pair_organoids=int(min_pair_organoids)
        self.n_splines=n_splines if size_dependent else 1
        t=n_markers+1;k=self.n_splines
        self.weights=torch.zeros(n_targets,self.hidden_dim,k,dtype=torch.float64)
        self.gamma_logits=torch.zeros(n_targets,k,dtype=torch.float64)
        self.knots=torch.zeros(k+4,dtype=torch.float64)
        self.log_variance=torch.zeros(n_targets,dtype=torch.float64)
        for name,value in dict(log_alpha=np.full((t,t),np.log(fixed_alpha)),
                active_pairs=np.zeros((t,t),bool),pair_support=np.zeros((t,t),int),
                target_scale=np.ones(n_targets),reference_design=np.zeros((t,self.radius,t))).items():
            self.register_buffer(name,torch.as_tensor(value))

    def basis(self,N):
        if not self.size_dependent:return np.ones((np.asarray(N).size,1))
        return super().basis(N)

    def configure(self,samples):
        # The parent establishes identical coefficient contrasts for all models.
        if not self.size_dependent:
            self.n_splines=4;self.knots=torch.zeros(8,dtype=torch.float64)
        super().configure(samples)
        if not self.size_dependent:
            self.n_splines=1;self.knots=torch.zeros(5,dtype=torch.float64)
        t=self.n_markers+1
        scale=np.sqrt(np.mean([np.mean(np.asarray(s['y']).reshape(-1,self.n_targets)**2,axis=0) for s in samples],axis=0))
        self.target_scale.copy_(torch.as_tensor(np.maximum(scale,1e-8)))
        support=np.zeros((t,t),int)
        for s in samples:
            totals=s['counts'][:,:self.radius].sum(axis=1) if self.radius else np.zeros((len(s['identity']),t))
            for a in range(t):
                take=s['identity']==a
                if take.any():support[a]+=np.any(totals[take]>0,axis=0)
        self.pair_support.copy_(torch.as_tensor(support));self.active_pairs.copy_(torch.as_tensor((support>=self.min_pair_organoids)&(self.activation=='learned')))
        self.refresh_reference(samples)
        return self

    def alpha(self):
        return np.exp(self._array(self.log_alpha)) if self.activation!='linear' else np.zeros_like(self._array(self.log_alpha))

    def raw_response(self,identities,counts,derivative=False):
        c=np.asarray(counts,dtype=float)[:,:self.radius];totals=c.sum(axis=1);population=totals.sum(axis=1,keepdims=True)
        fraction=np.divide(totals,population,out=np.zeros_like(totals),where=population>0)
        share=np.divide(c,totals[:,None,:],out=np.zeros_like(c),where=totals[:,None,:]>0)
        f,df=normalized_tanh(fraction,self.alpha()[identities],True)
        value=f[:,None,:]*share
        return (value,df[:,None,:]*share) if derivative else value

    def refresh_reference(self,samples):
        t=self.n_markers+1;ref=np.zeros((t,self.radius,t));dref=ref.copy();den=np.zeros(t)
        for s in samples:
            raw,dr=self.raw_response(s['identity'],s['counts'],True)
            for a in np.unique(s['identity']):
                take=s['identity']==a
                ref[a]+=raw[take].mean(axis=0);dref[a]+=dr[take].mean(axis=0);den[a]+=1
        ref/=np.maximum(den[:,None,None],1);dref/=np.maximum(den[:,None,None],1)
        self.reference_design.copy_(torch.as_tensor(ref))
        return dref

    def response_features(self,identities,counts,centered=True):
        raw=self.raw_response(identities,counts)
        return raw-self._array(self.reference_design)[identities] if centered else raw

    def gamma(self,N):
        b=self.basis(N)
        return self.gamma_max*expit(b@self._array(self.gamma_logits).T) if self.coupling else np.zeros((len(b),self.n_targets))

    def predict_sample(self,sample,N=None):
        if not self.is_fitted:raise RuntimeError('Model not fitted.')
        n=sample['N'] if N is None else N;b=self.basis([n])[0]
        z=self.local_design(sample['identity'],sample['counts'])
        local=z@np.einsum('thk,k->ht',self._array(self.weights),b)
        result=local.copy()
        for target,g in enumerate(self.gamma([n])[0]):
            if g:result[:,target]=splu((eye(len(z),format='csc')-g*sample['transition']).tocsc()).solve((1-g)*local[:,target])
        return result*self._array(self.target_scale)

    def coefficients(self,N):
        b=self.basis(N);t=self.n_markers+1;d=t-1
        w=np.einsum('nk,ohk->noh',b,self._array(self.weights))*self._array(self.target_scale)[None,:,None]
        pairs=np.einsum('ad,norde,rbe->norab',self._array(self.center_contrast),w[:,:,t:].reshape(len(b),self.n_targets,self.radius,d,d),self._array(self.source_contrast))
        return dict(a=w[:,:,:t],P=pairs,gamma=self.gamma(N),alpha=self.alpha())

    def forward(self,x,edge_index,data=None):
        ids=fate_identities(x);edge=self._array(edge_index).astype(int)
        batch=getattr(data,'batch',None)
        batch=self._array(batch).astype(int) if batch is not None else np.zeros(len(x),int)
        if edge.size and np.any(batch[edge[0]]!=batch[edge[1]]):raise ValueError('Cross-graph edges.')
        supplied=getattr(data,'full_num_cells',None)
        sizes=np.bincount(batch) if supplied is None else np.asarray(self._array(supplied) if torch.is_tensor(supplied) else supplied).reshape(-1)
        predictions=np.zeros((len(x),self.n_targets))
        for index,n in enumerate(sizes):
            nodes=np.flatnonzero(batch==index);mapping=np.full(len(x),-1);mapping[nodes]=np.arange(len(nodes))
            subedge=mapping[edge[:,batch[edge[0]]==index]]
            s=dict(N=n,identity=ids[nodes],counts=exact_hop_counts(x[nodes],subedge,self.radius),transition=neighbor_average_matrix(len(nodes),subedge))
            predictions[nodes]=self.predict_sample(s)
        mu=torch.as_tensor(predictions,device=x.device)
        if self.n_targets==1:mu=mu[:,0]
        return (mu,self.log_variance.to(x.device).expand_as(mu)),torch.as_tensor(predictions,device=x.device)

    def contributions(self,sample,N=None):
        """Physical local and propagated components for every predicted target."""
        n=sample['N'] if N is None else N;co=self.coefficients([n]);ids=sample['identity']
        q=self.response_features(ids,sample['counts'])
        center=co['a'][0][:,ids].T
        pairs=np.stack([q*co['P'][0,j].transpose(1,0,2)[ids] for j in range(self.n_targets)],axis=1)
        spread_center=np.zeros_like(center);spread_pairs=np.zeros_like(pairs)
        for j,g in enumerate(co['gamma'][0]):
            columns=np.c_[center[:,j],pairs[:,j].reshape(len(ids),-1)]
            spread=splu((eye(len(ids),format='csc')-g*sample['transition']).tocsc()).solve((1-g)*columns) if g else columns
            spread_center[:,j]=spread[:,0];spread_pairs[:,j]=spread[:,1:].reshape(pairs[:,j].shape)
        return dict(local_center=center,local_pairs=pairs,local=center+pairs.sum(axis=(2,3)),
            propagated_center=spread_center,propagated_pairs=spread_pairs,prediction=spread_center+spread_pairs.sum(axis=(2,3)))
