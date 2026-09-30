"""Profiled fitting and training-only discrete distance-sign selection."""
import copy
import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
from src.models.curvature_energy import MeanCurvatureEnergy, graph_laplacian


class EnergyObjective:
    def __init__(self,model,samples,*,ridge=1e-4,pair_ridge=1e-3,slope_ridge=1e-3,lambda_ridge=1e-4,alpha_ridge=1e-5):
        self.model=model;self.samples=samples;self.lambda_ridge=lambda_ridge;self.alpha_ridge=alpha_ridge
        t=model.n_markers+1;self.penalty=np.full(model.hidden_dim,pair_ridge);self.penalty[:t]=ridge
        if model.size_dependent and model.pairs:self.penalty[t+1::2]+=slope_ridge
        self.ng=model.n_basis if model.accommodation else 0
        self.active=model.array(model.active_pairs).astype(bool)&(model.activation=='learned')
        self.basis=model.basis([s['N'] for s in samples]);self.evaluations=0
        for s in samples:
            if 'laplacian' not in s:s['laplacian']=graph_laplacian(s['transition'])

    def pack(self):
        m=self.model;return np.r_[m.array(m.lambda_logits) if self.ng else [],m.array(m.log_alpha)[self.active]]

    def unpack(self,p):
        m=self.model
        if self.ng:m.lambda_logits.copy_(torch.tensor(p[:self.ng]))
        a=m.array(m.log_alpha).copy();a[self.active]=p[self.ng:];m.log_alpha.copy_(torch.tensor(a))
        return m.refresh_reference(self.samples)

    def __call__(self,p):
        m=self.model;dref=self.unpack(p);dim=m.hidden_dim;M=len(self.samples)
        gram=np.diag(self.penalty);rhs=np.zeros(dim);cache=[];lam=m.strength([s['N'] for s in self.samples]);scale=float(m.target_scale)
        for s,strength in zip(self.samples,lam):
            z=m.design(s);solver=splu(eye(len(z),format='csc')+strength*s['laplacian']) if strength else None
            x=solver.solve(z) if solver else z;y=s['y']/scale;weight=1/(M*len(z))
            gram+=weight*x.T@x;rhs+=weight*x.T@y;cache.append((s,x,solver,y,weight))
        theta=cho_solve(cho_factor(gram,check_finite=False),rhs,check_finite=False);m.weights.copy_(torch.tensor(theta));m.is_fitted.fill_(True)
        loss=float(np.dot(self.penalty*theta,theta));gl=np.zeros(self.ng);ga=np.zeros_like(self.active,dtype=float)
        co=m.coefficients([s['N'] for s in self.samples])['hop1']/scale
        for index,(s,x,solver,y,weight) in enumerate(cache):
            prediction=x@theta;resid=prediction-y;loss+=weight*np.dot(resid,resid)
            adj=solver.solve(resid) if solver else resid
            if self.ng:
                derivative=-2*weight*np.dot(adj,s['laplacian']@prediction)+2*self.lambda_ridge*lam[index]/M
                gl+=derivative*expit(self.basis[index]@m.array(m.lambda_logits))*self.basis[index]
            if self.active.any():
                ids=s['identity'];_,dr=m.raw_response(s,True);dr-=dref[ids]
                local=co[index][ids]*(dr[:,0]+.5*m.array(m.signs)[ids]*dr[:,1])
                np.add.at(ga,ids,2*weight*adj[:,None]*local)
        loss+=self.lambda_ridge*np.mean(lam**2)
        if self.active.any():
            delta=p[self.ng:]-np.log(m.fixed_alpha);loss+=self.alpha_ridge*np.mean(delta**2)
            ga[self.active]+=2*self.alpha_ridge*delta/len(delta)
        self.evaluations+=1
        return float(loss),np.r_[gl,ga[self.active]]

    def sign_statistics(self):
        """Unconstrained propagated Gram statistics, independent of hop signs."""
        m=self.model
        gram=None;rhs=None;constant=0.;M=len(self.samples);scale=float(m.target_scale)
        for s,lam in zip(self.samples,m.strength([s['N'] for s in self.samples])):
            raw=m.raw_design(s);x=splu(eye(len(raw),format='csc')+lam*s['laplacian']).solve(raw) if lam else raw
            y=s['y']/scale;weight=1/(M*len(y))
            if gram is None:gram=np.zeros((x.shape[1],x.shape[1]));rhs=np.zeros(x.shape[1])
            gram+=weight*x.T@x;rhs+=weight*x.T@y;constant+=weight*y@y
        return gram,rhs,constant

    def select_signs(self,max_passes=20):
        """Same sequential CPU coordinate search for both numerical backends."""
        m=self.model
        if not m.pairs:return dict(flips=0,passes=0,coordinate_optimum=True)
        gram,rhs,constant=self.sign_statistics()
        def score(signs):
            transform=m.mapping(signs);b=transform.T@rhs;g=transform.T@gram@transform+np.diag(self.penalty)
            w=cho_solve(cho_factor(g,check_finite=False),b,check_finite=False)
            return float(constant-b@w)
        signs=m.array(m.signs).copy();best=score(signs);flips=0;eligible=np.argwhere(m.array(m.active_pairs))
        for step in range(max_passes):
            changed=0
            for a,b in eligible:
                signs[a,b]*=-1;candidate=score(signs)
                if candidate<best-1e-9:best=candidate;changed+=1;flips+=1
                else:signs[a,b]*=-1
            if not changed:
                m.signs.copy_(torch.tensor(signs));return dict(flips=flips,passes=step+1,coordinate_optimum=True)
        raise RuntimeError('Distance-sign search failed to reach coordinate optimum')


class TensorEnergyObjective(EnergyObjective):
    """Device-resident profiled objective with the same analytic envelope gradient.

    SciPy retains the small bounded nonlinear search and sequential sign choices.
    Graph solves, reference centering, dense statistics, linear profiling and
    gradient accumulation use double-precision PyTorch on the requested device.
    """
    def __init__(self,model,samples,*,device='cuda',solver_rtol=1e-11,**penalties):
        super().__init__(model,samples,**penalties)
        from src.models.energy_ops import EnergyBatch
        self.batch=EnergyBatch(samples,model.n_markers,device=device)
        self.solver_rtol=solver_rtol
        self.tensor_penalty=self.batch.tensor(self.penalty)
        self.tensor_basis=self.batch.tensor(self.basis)
        self.node_basis=self.tensor_basis[self.batch.graph_id]
        self.y=self.batch.tensor(np.concatenate([s['y'] for s in samples]))/self.batch.tensor(model.target_scale)
        self.last_solve={}

    @torch.no_grad()
    def __call__(self,p):
        m=self.model;b=self.batch
        if self.ng:m.lambda_logits.copy_(torch.as_tensor(p[:self.ng],device=m.lambda_logits.device))
        alpha=m.array(m.log_alpha).copy();alpha[self.active]=p[self.ng:]
        m.log_alpha.copy_(torch.as_tensor(alpha,device=m.log_alpha.device))
        response,derivative=b.responses(m)
        reference=b.reference(response);dref=b.reference(derivative)
        m.reference.copy_(reference.to(m.reference.device))
        design=b.design(m,response)
        solver,strength=b.solver(m,self.tensor_basis,rtol=self.solver_rtol)
        x=solver(design) if solver is not None else design
        gram=x.T@(b.weight[:,None]*x)+torch.diag(self.tensor_penalty)
        rhs=x.T@(b.weight*self.y)
        theta=torch.cholesky_solve(rhs[:,None],torch.linalg.cholesky(gram))[:,0]
        m.weights.copy_(theta.to(m.weights.device));m.is_fitted.fill_(True)
        prediction=x@theta;residual=prediction-self.y
        loss=(self.tensor_penalty*theta.square()).sum()+(b.weight*residual.square()).sum()
        adjoint=solver(residual) if solver is not None else residual
        gradients=[]
        if self.ng:
            lap_prediction=torch.sparse.mm(b.laplacian,prediction[:,None])[:,0]
            sigmoid=torch.sigmoid(self.tensor_basis@b.tensor(m.lambda_logits))
            gl=self.node_basis.T@(-2*b.weight*adjoint*lap_prediction*sigmoid[b.graph_id])
            gl+=self.tensor_basis.T@(2*self.lambda_ridge*strength*sigmoid/b.n_graphs)
            gradients.append(gl)
        if self.active.any():
            table=(b.tensor(m.contrast)@theta[b.n_types:].reshape(-1,m.n_basis)).reshape(b.n_types,b.n_types,m.n_basis)
            coefficient=(table[b.ids]*self.node_basis[:,None,:]).sum(2)
            dr=derivative-dref[b.ids]
            local=coefficient*(dr[:,0]+.5*b.tensor(m.signs)[b.ids]*dr[:,1])
            ga=b.center.T@(2*b.weight[:,None]*adjoint[:,None]*local)
            mask=b.tensor(self.active,torch.bool)
            delta=b.tensor(p[self.ng:])-np.log(m.fixed_alpha)
            gradients.append(ga[mask]+2*self.alpha_ridge*delta/len(delta))
            loss+=self.alpha_ridge*delta.square().mean()
        loss+=self.lambda_ridge*strength.square().mean()
        self.evaluations+=1
        self.last_solve={} if solver is None else dict(iterations=solver.last_iterations,relative_residual=solver.last_relative_residual)
        gradient=torch.cat(gradients) if gradients else b.tensor([])
        return float(loss),gradient.cpu().numpy()

    @torch.no_grad()
    def sign_statistics(self):
        b=self.batch;m=self.model
        design=b.design(m,expanded=True)
        solver,_=b.solver(m,self.tensor_basis,rtol=self.solver_rtol)
        x=solver(design) if solver is not None else design
        gram=x.T@(b.weight[:,None]*x);rhs=x.T@(b.weight*self.y)
        constant=(b.weight*self.y.square()).sum()
        return gram.cpu().numpy(),rhs.cpu().numpy(),float(constant)


def _energy_device(device):
    device=torch.device(device)
    if device.type not in ('cpu','cuda'):raise ValueError('Energy fitting supports CPU or CUDA')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; select device=cpu')
    return device


def fit_energy_partition(model,samples,penalty,*,max_iterations=250,tolerance=2e-5,max_sign_rounds=8,callback=None,device='cpu',solver_rtol=1e-11):
    device=_energy_device(device)
    obj=EnergyObjective(model,samples,**penalty) if device.type=='cpu' else TensorEnergyObjective(model,samples,device=device,solver_rtol=solver_rtol,**penalty)
    history=[]
    def continuous():
        start=obj.pack()
        if not len(start):
            value,_=obj(start);return dict(success=True,objective=value,gradient_max=0.,iterations=0)
        bounds=[(None,None)]*obj.ng+[(np.log(.02),np.log(100.))]*(len(start)-obj.ng)
        result=minimize(obj,start,method='L-BFGS-B',jac=True,bounds=bounds,
            options=dict(maxiter=max_iterations,ftol=1e-13,gtol=tolerance,maxls=35,maxcor=30))
        value,gradient=obj(result.x)
        def norm(p,g):
            g=g.copy()
            for j in range(obj.ng,len(p)):
                if (p[j]<=bounds[j][0]+1e-5 and g[j]>0) or (p[j]>=bounds[j][1]-1e-5 and g[j]<0):g[j]=0
            return float(np.max(abs(g)))
        error=norm(result.x,gradient)
        if error>max(5*tolerance,1e-4):
            result=minimize(obj,result.x,method='SLSQP',jac=True,bounds=bounds,options=dict(maxiter=max_iterations,ftol=1e-12))
            value,gradient=obj(result.x);error=norm(result.x,gradient)
        return dict(success=bool(np.isfinite(value) and error<=max(5*tolerance,1e-4)),objective=value,gradient_max=error,iterations=int(result.nit),message=str(result.message))
    for round_ in range(max_sign_rounds):
        if callback:callback(f'Continuous fit / sign round {round_+1}')
        info=continuous()
        if not info['success']:raise RuntimeError(f'Continuous fit did not converge: {info}')
        sign_info=obj.select_signs();history.append(dict(**info,**sign_info))
        if callback:callback(f'Sign round {round_+1}: {sign_info["flips"]} flips')
        if sign_info['flips']==0:
            info.update(sign_rounds=history,sign_coordinate_optimum=True,evaluations=obj.evaluations,device=str(device),solver_rtol=solver_rtol if device.type!='cpu' else None)
            return info
    raise RuntimeError('Alternating sign/continuous fit has not stabilized')


def energy_mse(model,samples,*,device='cpu'):
    predictions=model.predict_samples(samples,device=device)
    return np.asarray([np.mean((prediction-s['y'])**2) for prediction,s in zip(predictions,samples)])


def fit_energy(samples,*,model_settings,penalties,seed=42,inner_fraction=.2,starts=((.15,3.,1),(.8,10.,-1)),
               max_iterations=250,tolerance=2e-5,blas_threads=4,callback=None,device='cpu',solver_rtol=1e-11):
    device=_energy_device(device)
    order=np.random.default_rng(seed).permutation(len(samples));nv=max(1,round(inner_fraction*len(samples)))
    train=[samples[i] for i in order[nv:]];val=[samples[i] for i in order[:nv]];trials=[];best=None
    with threadpool_limits(limits=blas_threads):
        for pi,penalty in enumerate(penalties):
            fits=[]
            for path,(strength,alpha,sign) in enumerate(starts if model_settings.get('pairs',True) or model_settings.get('accommodation',True) else starts[:1]):
                if callback:callback(f'Inner penalty {pi+1}/{len(penalties)}, start {path+1}')
                m=MeanCurvatureEnergy(**model_settings).configure(train)
                m.lambda_logits[0]=np.log(np.expm1(strength));m.log_alpha.fill_(np.log(alpha if m.activation=='learned' else m.fixed_alpha))
                m.log_alpha[~m.active_pairs]=np.log(m.fixed_alpha)
                m.signs[m.active_pairs]=sign
                try:info=fit_energy_partition(m,train,penalty,max_iterations=max_iterations,tolerance=tolerance,callback=callback,device=device,solver_rtol=solver_rtol)
                except torch.cuda.OutOfMemoryError:
                    raise
                except RuntimeError as error:
                    trials.append(dict(penalty_index=pi,path=path,stage='inner',success=False,error=str(error)));continue
                mse=float(energy_mse(m,val,device=device).mean());trials.append(dict(penalty_index=pi,path=path,stage='inner',inner_mse=mse,**info))
                fits.append((info['objective'],m,info,mse))
            if fits:
                _,m,info,mse=min(fits,key=lambda row:row[0])
                if best is None or mse<best['inner_mse']:best=dict(inner_mse=mse,penalty=penalty,penalty_index=pi,states=[copy.deepcopy(f[1].state_dict()) for f in fits])
        if best is None:raise RuntimeError(f'All inner fits failed: {trials}')
        final=[]
        for path,state in enumerate(best.pop('states')):
            m=MeanCurvatureEnergy(**model_settings).configure(samples)
            # N basis has a new training reference: preserve the initialized function.
            shift=float(m.log_reference-state['log_reference'])
            m.lambda_logits.copy_(state['lambda_logits'])
            if m.size_dependent:m.lambda_logits[0]+=shift*m.lambda_logits[1]
            m.log_alpha.copy_(state['log_alpha']);m.signs.copy_(state['signs']);m.signs[~m.active_pairs]=1
            m.log_alpha[~m.active_pairs]=np.log(m.fixed_alpha)
            try:info=fit_energy_partition(m,samples,best['penalty'],max_iterations=max_iterations,tolerance=tolerance,callback=callback,device=device,solver_rtol=solver_rtol)
            except torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as error:
                trials.append(dict(path=path,stage='outer',success=False,error=str(error)));continue
            trials.append(dict(path=path,stage='outer',**info));final.append((info['objective'],m,info))
        if not final:raise RuntimeError(f'All outer fits failed: {trials}')
        _,model,info=min(final,key=lambda row:row[0]);model.log_variance.fill_(np.log(max(float(energy_mse(model,samples,device=device).mean()),1e-30)))
    metadata=dict(device=str(device),solver_rtol=solver_rtol if torch.device(device).type!='cpu' else None,best=best,optimizer=info,inner_train=[s['organoid_str'] for s in train],inner_validation=[s['organoid_str'] for s in val],fit_organoids=[s['organoid_str'] for s in samples],sign_selection='Training-only multistart alternating continuous/coordinate optimization; no global-optimum guarantee')
    return model,metadata,pd.DataFrame(trials)
