"""Profiled least-squares fitting of explicit fate terms and smooth coupling.

Predicted mode solves graph predictions self-consistently. Measured mode is an
explicit conditional task using observed neighbor curvature (excluding the center). For each coupling curve, solve the linear fate coefficients exactly;
optimize the coupling spline using the envelope gradient through sparse solves.
"""
import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logit
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
from src.models.coupled_fate import CoupledFateCurvature, neighbor_average_matrix
from src.data.neighborhood_counts import fate_identities, exact_hop_counts


def graph_samples(graphs,radius):
    result=[]
    for g in graphs:
        y=g.y.detach().cpu().numpy().reshape(-1).astype(float)
        if len(y)!=len(g.x) or not np.isfinite(y).all():raise ValueError('Finite scalar targets required.')
        result.append(dict(organoid_str=str(g.organoid_str),N=len(g.x),identity=fate_identities(g.x),
            counts=exact_hop_counts(g.x,g.edge_index,radius),transition=neighbor_average_matrix(len(g.x),g.edge_index),y=y))
    if not result or len({s['organoid_str'] for s in result})!=len(result):raise ValueError('Nonempty unique organoids required.')
    return result


class ProfileObjective:
    """Reusable objective for one training partition, basis and penalty choice."""
    def __init__(self,model,samples,*,ridge,pair_ridge,smoothness,gamma_smoothness,gamma_ridge):
        if any(not np.isfinite(v) or v<0 for v in [ridge,pair_ridge,smoothness,gamma_smoothness,gamma_ridge]):
            raise ValueError('Penalties must be finite and nonnegative.')
        self.model,self.samples=model,samples
        self.local=[model.local_design(s['identity'],s['counts']) for s in samples]
        self.basis=model.basis([s['N'] for s in samples]);k=model.n_splines
        second=np.diff(np.eye(k),n=2,axis=0);self.second=second.T@second
        self.penalty=smoothness*np.kron(np.eye(model.hidden_dim),self.second)
        diagonal=np.full(model.hidden_dim,pair_ridge,dtype=float);diagonal[:model.n_markers+1]=ridge
        self.penalty.flat[::len(self.penalty)+1]+=np.repeat(diagonal,k)+1e-10
        self.gamma_smoothness,self.gamma_ridge=gamma_smoothness,gamma_ridge
        self.measured=[model.measured_neighbors(s) for s in samples] if model.neighbor_curvature=='measured' else None
        self.evaluations=0

    def __call__(self,eta):
        model=self.model;model.gamma_logits.copy_(torch.as_tensor(eta))
        gammas=model.gamma([s['N'] for s in self.samples]);p=model.weights.numel()
        gram=np.zeros((p,p));rhs=np.zeros(p);solvers=[];spread=[]
        for index,(s,z,b,g) in enumerate(zip(self.samples,self.local,self.basis,gammas)):
            solver=splu((eye(len(z),format='csc')-g*s['transition']).tocsc()) if g and self.measured is None else None
            if self.measured is not None:
                neighbor,available=self.measured[index]
                x=(1-g*available[:,None])*z
                target=s['y']-g*neighbor
            else:
                x=solver.solve((1-g)*z) if solver else z
                target=s['y']
            weight=1/(len(self.samples)*len(z))
            gram+=np.kron(x.T@x,np.outer(b,b))*weight;rhs+=np.kron(x.T@target,b)*weight
            solvers.append(solver);spread.append(x)
        theta=cho_solve(cho_factor(gram+self.penalty,lower=True,check_finite=False),rhs,check_finite=False)
        model.weights.copy_(torch.tensor(theta.reshape(model.weights.shape)));model.is_fitted.fill_(True)
        loss=float(theta@self.penalty@theta);gradient=np.zeros(model.n_splines)
        for index,(s,z,x,b,g,solver) in enumerate(zip(self.samples,self.local,spread,self.basis,gammas,solvers)):
            coef=theta.reshape(model.weights.shape)@b;prediction=x@coef
            if self.measured is not None:prediction=prediction+g*self.measured[index][0]
            residual=prediction-s['y']
            loss+=np.mean(residual**2)/len(self.samples)
            if solver is not None or self.measured is not None:
                derivative=(self.measured[index][0]-self.measured[index][1]*(z@coef)) if self.measured is not None else solver.solve(s['transition']@prediction-z@coef)
                chain=g*(1-g/model.gamma_max)*b
                gradient+=2*np.mean(residual*derivative)*chain/len(self.samples)
                gradient+=2*self.gamma_ridge*g*chain/len(self.samples)
        if model.coupling:
            loss+=self.gamma_smoothness*(eta@self.second@eta)+self.gamma_ridge*np.mean(gammas**2)
            gradient+=2*self.gamma_smoothness*self.second@eta
        self.evaluations+=1
        return loss,gradient


def fit_partition(model,samples,*,ridge,pair_ridge,smoothness,gamma_smoothness=.01,gamma_ridge=1e-4,
                  gamma_start=.2,max_iterations=50,tolerance=1e-7,callback=None):
    """Fit one configured partition; return explicit convergence diagnostics."""
    objective=ProfileObjective(model,samples,ridge=ridge,pair_ridge=pair_ridge,smoothness=smoothness,
        gamma_smoothness=gamma_smoothness,gamma_ridge=gamma_ridge)
    eta=np.full(model.n_splines,logit(np.clip(gamma_start/model.gamma_max,1e-5,1-1e-5)))
    if model.coupling:
        def update(x):
            if callback:callback(f'Coupling optimizer: {objective.evaluations} objective evaluations')
        # Scale the WHOLE objective, including penalties. This leaves its
        # minimizer unchanged but avoids premature ftol stopping on curvature².
        objective_scale=max(float(np.mean([np.mean(s['y']**2) for s in samples])),1e-12)
        def scaled_objective(x):
            value,gradient=objective(x)
            return value/objective_scale,gradient/objective_scale
        # Whiten the stiff spline-smoothness directions in the initial inverse
        # Hessian. Gradient convergence is required; tiny first steps must not
        # be mistaken for convergence of a still-flat size-dependent curve.
        hessian=2*gamma_smoothness*objective.second/objective_scale+.01*np.eye(model.n_splines)
        inverse=np.linalg.inv(hessian);inverse=(inverse+inverse.T)/2
        result=minimize(scaled_objective,eta,jac=True,method='BFGS',
            options=dict(maxiter=max_iterations,gtol=tolerance,hess_inv0=inverse),callback=update)
        value,grad=objective(result.x)
        info=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
            evaluations=objective.evaluations,objective=float(value),gradient_norm=float(np.linalg.norm(grad)),objective_scale=objective_scale,method='preconditioned BFGS',scaled_gradient_norm=float(np.linalg.norm(grad)/objective_scale))
    else:
        value,_=objective(eta)
        info=dict(success=True,message='Exact uncoupled linear solve',iterations=0,evaluations=1,objective=float(value),gradient_norm=0.)
    return info


def sample_mse(model,samples):
    return np.array([np.mean((model.predict_sample(s)-s['y'])**2) for s in samples])


def fit_coupled_fate(samples,*,model_settings,penalties,inner_fraction=.2,seed=42,
                     gamma_starts=(.15,.6),gamma_smoothness=.01,gamma_ridge=1e-4,
                     max_iterations=50,tolerance=1e-7,blas_threads=2,require_convergence=True,callback=None):
    """Select penalties on an inner organoid holdout, then refit outer training.

    Coupling starts are selected by training objective within each penalty trial,
    never by outer validation. Nonconverged candidates are recorded and excluded
    when require_convergence=True. The final refit must then converge as well.
    """
    if len(samples)<4 or not 0<inner_fraction<1 or not penalties or not gamma_starts:
        raise ValueError('Need >=4 organoids, nonempty grids and 0<inner_fraction<1.')
    if blas_threads<1 or max_iterations<1 or tolerance<=0:raise ValueError('Invalid optimizer settings.')
    if any(not 0<g<model_settings.get('gamma_max',.95) for g in gamma_starts):raise ValueError('Initial gamma must lie below gamma_max.')
    order=np.random.default_rng(seed).permutation(len(samples));nv=max(1,min(len(samples)-2,round(inner_fraction*len(samples))))
    train=[samples[i] for i in order[nv:]];val=[samples[i] for i in order[:nv]]
    trials=[];best=None
    with threadpool_limits(limits=blas_threads):
        for index,penalty in enumerate(penalties):
            if set(penalty)!={'ridge','pair_ridge','smoothness'}:raise ValueError('Each penalty entry needs ridge, pair_ridge and smoothness.')
            fitted=[]
            starts=gamma_starts if model_settings.get('coupling',True) else gamma_starts[:1]
            for start in starts:
                if callback:callback(f'Inner penalty {index+1}/{len(penalties)}, initial gamma={start:g}')
                model=CoupledFateCurvature(**model_settings).configure(train)
                info=fit_partition(model,train,**penalty,gamma_start=start,gamma_smoothness=gamma_smoothness,
                    gamma_ridge=gamma_ridge,max_iterations=max_iterations,tolerance=tolerance,callback=callback)
                mse=float(sample_mse(model,val).mean())
                trials.append(dict(penalty_index=index,gamma_start=start,inner_mse=mse,**penalty,**info))
                if info['success'] or not require_convergence:fitted.append((info['objective'],model,info,start,mse))
            if not fitted:continue
            _,model,info,start,mse=min(fitted,key=lambda x:x[0])
            if best is None or mse<best['inner_mse']:
                best=dict(inner_mse=mse,penalty=penalty,gamma_start=start,inner_optimizer=info,
                    gamma_at_training_median=float(model.gamma([np.median([s['N'] for s in train])])[0]))
        if best is None:raise RuntimeError('No inner fit converged; increase max_iterations or inspect penalty/optimizer settings.')
        if callback:callback('Refit selected penalty on the complete outer-training partition')
        final_fits=[]
        starts=[best['gamma_at_training_median'],*gamma_starts] if model_settings.get('coupling',True) else gamma_starts[:1]
        for start in dict.fromkeys(starts):
            model=CoupledFateCurvature(**model_settings).configure(samples)
            info=fit_partition(model,samples,**best['penalty'],gamma_start=max(start,1e-5),
                gamma_smoothness=gamma_smoothness,gamma_ridge=gamma_ridge,max_iterations=max_iterations,tolerance=tolerance,callback=callback)
            if info['success'] or not require_convergence:final_fits.append((info['objective'],model,info))
        if not final_fits:raise RuntimeError('Final fit did not converge; no model should be saved as complete.')
        _,model,info=min(final_fits,key=lambda x:x[0])
        train_mse=float(sample_mse(model,samples).mean());model.log_variance.fill_(np.log(max(train_mse,1e-30)))
    metadata=dict(best=best,optimizer=info,train_mse=train_mse,
        inner_train=[s['organoid_str'] for s in train],inner_validation=[s['organoid_str'] for s in val],
        fit_organoids=[s['organoid_str'] for s in samples],effective_reach=(f'{max(1,model.radius)} hops with observed neighbor targets' if model.neighbor_curvature=='measured' else 'whole connected component' if model.coupling else f'{model.radius} hops'),
        prediction=('conditional on observed neighbor curvature; not fate-only prediction' if model.neighbor_curvature=='measured' else 'self-consistent predicted neighbor curvature; no measured-neighbor input'),
        neighbor_curvature=model.neighbor_curvature,
        variance='constant training residual MSE; not calibrated uncertainty')
    return model,metadata,pd.DataFrame(trials)


class ResponseObjective:
    """Profile linear coefficients; differentiate propagation and saturation.

    Equal weight per organoid and per training-scaled target. Responses and their
    centering references are recomputed on training data only for every alpha.
    """
    def __init__(self,model,samples,*,ridge=1e-4,pair_ridge=1e-3,smoothness=.01,
                 gamma_smoothness=.1,gamma_ridge=1e-4,alpha_ridge=1e-5):
        self.model,self.samples=model,samples
        self.basis=model.basis([s['N'] for s in samples]);k=model.n_splines
        self.second=np.diff(np.eye(k),n=2,axis=0).T@np.diff(np.eye(k),n=2,axis=0)
        diagonal=np.full(model.hidden_dim,pair_ridge);diagonal[:model.n_markers+1]=ridge
        self.penalty=smoothness*np.kron(np.eye(model.hidden_dim),self.second)+np.diag(np.repeat(diagonal,k)+1e-10)
        self.gamma_smoothness,self.gamma_ridge,self.alpha_ridge=gamma_smoothness,gamma_ridge,alpha_ridge
        self.active=model._array(model.active_pairs).astype(bool)
        self.ng=model.n_targets*k if model.coupling else 0
        self.evaluations=0
        self.local_cache=None if self.active.any() else [model.local_design(s['identity'],s['counts']) for s in samples]

    def pack(self):
        m=self.model
        return np.r_[m._array(m.gamma_logits).ravel() if self.ng else [],m._array(m.log_alpha)[self.active]]

    def __call__(self,parameters):
        m=self.model;k=m.n_splines;t=m.n_markers+1;o=m.n_targets;M=len(self.samples)
        if self.ng:m.gamma_logits.copy_(torch.tensor(parameters[:self.ng].reshape(o,k)))
        if self.active.any():
            alpha=m._array(m.log_alpha).copy();alpha[self.active]=parameters[self.ng:];m.log_alpha.copy_(torch.tensor(alpha))
            dref=m.refresh_reference(self.samples)
        local=self.local_cache or [m.local_design(s['identity'],s['counts']) for s in self.samples]
        gammas=m.gamma([s['N'] for s in self.samples]);p=m.hidden_dim*k
        loss=0.;gg=np.zeros((o,k));ga=np.zeros((t,t));scale=m._array(m.target_scale)
        for target in range(o):
            gram=np.zeros((p,p));rhs=np.zeros(p);spread=[];solvers=[]
            for s,z,b,g in zip(self.samples,local,self.basis,gammas[:,target]):
                solver=splu((eye(len(z),format='csc')-g*s['transition']).tocsc()) if g else None
                x=solver.solve((1-g)*z) if solver else z
                y=np.asarray(s['y']).reshape(-1,o)[:,target]/scale[target];weight=1/(M*len(z))
                gram+=np.kron(x.T@x,np.outer(b,b))*weight;rhs+=np.kron(x.T@y,b)*weight
                spread.append(x);solvers.append(solver)
            theta=cho_solve(cho_factor(gram+self.penalty,check_finite=False),rhs,check_finite=False)
            coef=theta.reshape(m.hidden_dim,k);m.weights[target].copy_(torch.tensor(coef))
            loss+=theta@self.penalty@theta/o
            for s,z,x,b,g,solver in zip(self.samples,local,spread,self.basis,gammas[:,target],solvers):
                weights=coef@b;pred=x@weights;y=np.asarray(s['y']).reshape(-1,o)[:,target]/scale[target];resid=pred-y
                loss+=np.mean(resid**2)/(M*o)
                if self.ng:
                    derivative=solver.solve(s['transition']@pred-z@weights)
                    chain=g*(1-g/m.gamma_max)*b
                    gg[target]+=2*(np.mean(resid*derivative)+self.gamma_ridge*g)*chain/(M*o)
                if self.active.any():
                    adjoint=solver.solve((1-g)*resid,trans='T') if solver else resid
                    _,dr=m.raw_response(s['identity'],s['counts'],True);dr-=dref[s['identity']]
                    pair=np.einsum('ad,rde,rbe->arb',m._array(m.center_contrast),weights[t:].reshape(m.radius,t-1,t-1),m._array(m.source_contrast))
                    pernode=np.sum(dr*pair[s['identity']],axis=1)*adjoint[:,None]*2/(M*o*len(z))
                    np.add.at(ga,s['identity'],pernode)
        if self.ng:
            eta=m._array(m.gamma_logits)
            loss+=(self.gamma_smoothness*np.sum((eta@self.second)*eta)+self.gamma_ridge*np.mean(gammas**2)*o)/o
            gg+=2*self.gamma_smoothness*(eta@self.second)/o
        if self.active.any():
            delta=parameters[self.ng:]-np.log(m.fixed_alpha)
            loss+=self.alpha_ridge*np.mean(delta**2)
            ga[self.active]+=2*self.alpha_ridge*delta/len(delta)
        m.is_fitted.fill_(True);self.evaluations+=1
        return float(loss),np.r_[gg.ravel() if self.ng else [],ga[self.active]]


def fit_response_partition(model,samples,penalty,*,gamma_start=.2,alpha_start=3.,max_iterations=250,tolerance=2e-5,callback=None,initial_state=None):
    """Preconditioned profile BFGS with smoothly bounded saturation coordinates."""
    from scipy.special import expit
    model.gamma_logits.fill_(logit(gamma_start/model.gamma_max));model.log_alpha.fill_(np.log(alpha_start))
    if initial_state is not None:
        model.gamma_logits.copy_(torch.as_tensor(initial_state['gamma_logits']))
        model.log_alpha.copy_(torch.as_tensor(initial_state['log_alpha']))
    model.log_alpha[~model.active_pairs]=np.log(model.fixed_alpha)
    model.refresh_reference(samples)
    objective=ResponseObjective(model,samples,**penalty);start=objective.pack();ng=objective.ng
    if len(start):
        lo,hi=np.log(.02),np.log(100.)
        def decode(z):
            parameters=z.copy();v=expit(z[ng:]);parameters[ng:]=lo+(hi-lo)*v
            return parameters,(hi-lo)*v*(1-v)
        def function(z):
            parameters,chain=decode(z);value,gradient=objective(parameters);gradient[ng:]*=chain
            return value,gradient
        z=start.copy();z[ng:]=logit(np.clip((start[ng:]-lo)/(hi-lo),1e-6,1-1e-6))
        k=model.n_splines
        hessian=2*objective.gamma_smoothness*objective.second/model.n_targets+.02*np.eye(k)
        inverse=np.eye(len(start));block=np.linalg.inv(hessian)
        for j in range(model.n_targets if ng else 0):inverse[j*k:(j+1)*k,j*k:(j+1)*k]=block
        _,chain=decode(z)
        for j in range(ng,len(start)):inverse[j,j]=min(100/max(chain[j-ng]**2,1e-8),1e5)
        inverse=(inverse+inverse.T)/2
        def update(x):
            if callback:callback(f'Preconditioned profile solver: {objective.evaluations} evaluations')
        result=minimize(function,z,jac=True,method='BFGS',options=dict(maxiter=max_iterations,gtol=tolerance,hess_inv0=inverse),callback=update)
        parameters,_=decode(result.x);value,gradient=objective(parameters)
        def projected_norm(parameters,gradient):
            g=gradient.copy()
            for j in range(ng,len(g)):
                if (parameters[j]<=lo+1e-4 and g[j]>0) or (parameters[j]>=hi-1e-4 and g[j]<0):g[j]=0
            return float(np.max(np.abs(g)))
        norm=projected_norm(parameters,gradient);fallback=False
        if norm>max(tolerance*5,1e-4):
            fallback=True
            result=minimize(objective,parameters,jac=True,method='SLSQP',bounds=[(None,None)]*ng+[(lo,hi)]*(len(start)-ng),
                options=dict(maxiter=max_iterations,ftol=1e-12),callback=update)
            parameters=result.x;value,gradient=objective(parameters);norm=projected_norm(parameters,gradient)
        info=dict(success=bool(np.isfinite(value) and norm<=max(tolerance*5,1e-4)),fallback=fallback,message=str(result.message),
            iterations=int(result.nit),evaluations=objective.evaluations,objective=value,gradient_max=norm,
            method='preconditioned profile BFGS with smooth saturation bounds',warm_start=initial_state is not None)
    else:
        value,_=objective(start);info=dict(success=True,message='Exact linear solve',iterations=0,evaluations=1,objective=value,gradient_max=0.)
    return info


def response_mse(model,samples,scaled=False):
    errors=np.array([np.mean((model.predict_sample(s)-np.asarray(s['y']).reshape(-1,model.n_targets))**2,axis=0) for s in samples])
    return errors/model._array(model.target_scale)**2 if scaled else errors


def fit_fate_response(samples,*,model_settings,penalties,seed=42,inner_fraction=.2,
                      starts=((.15,1.),(.55,8.)),max_iterations=250,tolerance=2e-5,blas_threads=4,callback=None):
    """Training-only continuation paths, inner penalty selection and outer refit.

    Independent starts define optimization paths through the penalty grid. Each
    converged path warm-starts the next penalty. The chosen penalty's converged
    inner solutions initialize the outer refits; no outer target selects a path.
    """
    from src.models.coupled_fate import CoupledFateResponse
    order=np.random.default_rng(seed).permutation(len(samples));nv=max(1,round(len(samples)*inner_fraction))
    train=[samples[i] for i in order[nv:]];val=[samples[i] for i in order[:nv]];trials=[];best=None;warm_paths={}
    scales=np.sqrt(np.mean([np.mean(np.asarray(s['y']).reshape(-1,model_settings['n_targets'])**2,axis=0) for s in train],axis=0))
    def state(model):return dict(gamma_logits=model.gamma_logits.tolist(),log_alpha=model.log_alpha.tolist())
    active_starts=starts if model_settings['coupling'] or model_settings['activation']=='learned' else starts[:1]
    with threadpool_limits(limits=blas_threads):
        for index,penalty in enumerate(penalties):
            fits=[]
            for path,(gs,as_) in enumerate(active_starts):
                if callback:callback(f'Penalty {index+1}/{len(penalties)}; path {path+1}/{len(active_starts)}')
                model=CoupledFateResponse(**model_settings).configure(train)
                info=fit_response_partition(model,train,penalty,gamma_start=gs,alpha_start=as_,initial_state=warm_paths.get(path),
                    max_iterations=max_iterations,tolerance=tolerance,callback=callback)
                mse=float(np.mean(response_mse(model,val)/scales**2))
                trials.append(dict(penalty_index=index,path=path,gamma_start=gs,alpha_start=as_,inner_mse=mse,**info))
                if info['success']:
                    warm_paths[path]=state(model);fits.append((info['objective'],model,info,mse,gs,as_))
            if fits:
                _,model,info,mse,gs,as_=min(fits,key=lambda a:a[0])
                if best is None or mse<best['inner_mse']:
                    best=dict(inner_mse=mse,penalty=penalty,penalty_index=index,gamma_start=gs,alpha_start=as_,
                        inner_states=[state(fit[1]) for fit in fits])
        if best is None:raise RuntimeError(f'No converged inner fit: {trials}')
        fits=[]
        for path,initial in enumerate(best['inner_states']):
            if callback:callback(f'Outer-training warm refit {path+1}/{len(best["inner_states"])}')
            model=CoupledFateResponse(**model_settings).configure(samples)
            info=fit_response_partition(model,samples,best['penalty'],initial_state=initial,
                max_iterations=max_iterations,tolerance=tolerance,callback=callback)
            trials.append(dict(penalty_index=best['penalty_index'],path=path,inner_mse=None,stage='refit',**info))
            if info['success']:fits.append((info['objective'],model,info))
        if not fits:raise RuntimeError(f'No converged outer fit: {trials[-len(best["inner_states"]):]}')
        _,model,info=min(fits,key=lambda a:a[0]);mse=response_mse(model,samples).mean(axis=0)
        model.log_variance.copy_(torch.as_tensor(np.log(np.maximum(mse,1e-30))))
    metadata=dict(best=best,optimizer=info,train_mse=mse.tolist(),target_scale=model._array(model.target_scale).tolist(),
        inner_train=[s['organoid_str'] for s in train],inner_validation=[s['organoid_str'] for s in val],
        fit_organoids=[s['organoid_str'] for s in samples],prediction='Fate-only, predicted-neighbor propagation',
        parameterization='whole-neighborhood tanh activation times source distance share; training-centered responses')
    return model,metadata,pd.DataFrame(trials)
