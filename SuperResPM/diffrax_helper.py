# Standard library
import gc
from functools import partial
from collections.abc import Callable

# Typing
from typing import ClassVar
from typing_extensions import TypeAlias

# JAX & related
import jax
import jax.numpy as jnp
from jax import lax, tree
from jax.example_libraries.optimizers import adam
from jax.experimental.ode import odeint

# JAX-Cosmo
import jax_cosmo as jc
from jax_cosmo.background import *

# JAXPM
from jaxpm.pm import pm_forces, linear_field, lpt, make_ode_fn
from jaxpm.growth import (
    E,dGfa, dGf2a, growth_factor,
    growth_factor_second, growth_rate, growth_rate_second
)
from jaxpm.growth import E, growth_factor as Gp, Gf
from jaxpm.painting import cic_paint, cic_paint_2d, cic_paint_dx, cic_read
from jaxpm.distributed import uniform_particles

# Diffrax
import diffrax
from diffrax import (
    LocalLinearInterpolation, RESULTS, AbstractTerm,
    AbstractSolver, ODETerm, ConstantStepSize,
    SaveAt, diffeqsolve
)
from diffrax._custom_types import (
    Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
)
from diffrax._solver.base import AbstractReversibleSolver

# Equinox
import equinox as eqx
from equinox.internal import ω

# Jaxtyping
from jaxtyping import ArrayLike, Float, PyTree




##################
#FastPM ODE TERM #
##################

class FPMODE(ODETerm):
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        action = kwargs.get("action", "Drift") 
        cosmo = kwargs.get("cosmo", None)
        tc = kwargs.get("tc", 0.5)
        t0_in = jnp.abs(t0)
        t1_in = jnp.abs(t1)
        tc_in = jnp.abs(tc)
        if cosmo is None:
            return 0.0

        if action == "Drift":  # Drift case
            d1 = drift_factor_in(tc_in,t0_in, t1_in, cosmo)

            return d1

        elif action == "Kick":
            k1 = kick_factor_in(tc_in,t0_in, t1_in, cosmo)
            return k1
        else:
            raise ValueError(f"Unknown action type: {action}")

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = tuple[RealScalarLike, PyTree, RealScalarLike]

###################
#FastPM ODE Solver#
###################

class FPMLeapFrog(AbstractReversibleSolver):
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )
    initial_t0: RealScalarLike
    final_t1:RealScalarLike
    def init(
        self,
        terms: FPMODE,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return (t0, y0, t1-t0)
    def step(
        self,
        terms: FPMODE,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del made_jump
        t0 =  jnp.atleast_1d(t0)
        t1 =  jnp.atleast_1d(t1)
        tmid = (t0 + t1)/2
        initial_t0=args[2]
        cosmo = args[0]
        cosmo._workspace=args[1]
        args[-2]=0 # Recompute forces or not 
        args[-1]=0
        # First kick hal-step
        control1 = terms.contr(t0, tmid, tc=t0, action="Kick", cosmo=cosmo)
        yin1 = (y0**ω + terms.vf_prod(t0, y0, args, control1) ** ω).ω
        args[-1]=1
        args[-2]=0 # Recompute forces or not 
        # Drift full step
        #jax.debug.print("fyin1: {x}", x=[yin1[1,0,0,0], y0[1,0,0,0], control1, t0, tmid])
        control2 = terms.contr(t0, t1,tc=tmid, action="Drift", cosmo=cosmo)
        yin2= (yin1**ω + terms.vf_prod(tmid, yin1, args, control2) ** ω).ω
        #jax.debug.print("fyin2: {x}", x=[yin2[0,0,0,0], yin1[0,0,0,0], control2, t0, t1])
        del yin1
        args[-1]=0
        args[-2]=1 # Recompute forces or not 
        yin2= yin2.at[2].multiply(0) #jnp.zeros_like(yin2[-2])
        # 2nd kick
        control3 = terms.contr(tmid, t1, tc=t1, action="Kick", cosmo=cosmo)
        yin3 = (yin2**ω + terms.vf_prod(t1,yin2, args, control3) ** ω).ω
        yin3=yin3.at[2].divide(control3)
        #jax.debug.print("fyin3: {x}", x=[yin3[1,0,0,0], yin2[1,0,0,0], yin2[2,0,0,0], yin3[2,0,0,0], control3, tmid, t1])
        del yin2
        gc.collect()
        y1 = yin3
        dense_info = dict(y0=y0, y1=y1)
        solver_state=(t0[0],y0, t1[0]-t0[0])
        return y1, None, dense_info, solver_state, RESULTS.successful
    def backward_step(
        self,
        terms: FPMODE,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y2: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y,  DenseInfo, _SolverState]:
        t0, y1, dt = solver_state
        del made_jump
        #dt = t1-t0
        tm1 = t0-dt
        t0 =  jnp.atleast_1d(t0)
        tm1 =  jnp.atleast_1d(tm1)
        dt=jnp.atleast_1d(dt)
        tmid = (t0 + tm1)/2
        initial_t0=args[2]
        cosmo = args[0]
        cosmo._workspace=args[1]
    

        args[-1]=0
        args[-2]=0 # Recompute forces or not 
        #args[-2]=1 # Recompute forces or not 
        #y2= y2.at[2].multiply(0) #jnp.zeros_like(yin2[-2])
        # First kick hal-step
        control1 = terms.contr(t0, tmid, tc=t0, action="Kick", cosmo=cosmo)
#=jnp.zeros_like(y2[-2])
        yin1 = (y2**ω + terms.vf_prod(t0, y1, args, control1) ** ω).ω
        #jax.debug.print("yin1: {x}", x=[yin1[1,0,0,0], y2[1,0,0,0], yin1[2,0,0,0], control1, t0, tmid])
        args[-1]=1
        args[-2]=0 # Recompute forces or not 
        # Drift full step
        control2 = terms.contr(t0, tm1,tc=tmid, action="Drift", cosmo=cosmo)
        yin2 = (yin1**ω + terms.vf_prod(tmid,  yin1, args, control2) ** ω).ω
        #jax.debug.print("yin2: {x}", x=[yin2[0,0,0,0], yin1[0,0,0,0], control2, t0, tm1])
        del yin1
        args[-1]=0
        # 2nd kick
        args[-2]=1 # Recompute forces or not 
        control3 = terms.contr(tmid, tm1, tc=tm1, action="Kick", cosmo=cosmo)
        yin2=yin2.at[2].multiply(0) 
#jnp.zeros_like(yin2[-2])
        yin3 = (yin2**ω + terms.vf_prod(tm1, yin2, args, control3) ** ω).ω

        yin3=yin3.at[2].divide(control3)


        #jax.debug.print("yin3: {x}", x=[yin3[1,0,0,0], yin2[1,0,0,0], control3, tmid, tm1])
        del yin2
        gc.collect()
        y0 = yin3

        solver_state = jax.lax.cond(
            tm1[0] > 0, lambda _: (tm1[0], y0, dt[0]), lambda _: (t0[0],y1,dt[0]), None
        )
        dense_info = dict(y0=y1, y1=y2)
        return y0, dense_info, solver_state

    
    @partial(jax.jit, static_argnums=0)
    def func(
        self,
        terms: FPMODE,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        f1 = terms.vf(jnp.atleast_1d(t0), y0, [args[0], args[1], 1, 0])
        f2 = terms.vf(jnp.atleast_1d(t0), f1, [args[0], args[1], 0, 1])
        return jnp.stack([f2[0], f1[1], y0[2]], axis=0)
        
def symplectic_ode(mesh_shape, paint_absolute_pos=True, halo_size=0, sharding=None):
    def drift(a, vel, args):
        """
        state is a tuple (position, velocities)
        """
        a_in = jnp.abs(a)
        cosmo = args[0]
        cosmo._workspace= args[1]
        # Computes the update of position (drift)
        dpos = 1 / (a_in**3 * E(cosmo, a_in)) * vel

        return dpos
    
    def kick(a, pos, args, forces):
        """
        state is a tuple (position, velocities)
        """
        # Computes the update of velocity (kick)
        cosmo= args[0]
        cosmo._workspace= args[1]
        def forcefun(_): 
            return pm_forces(
                pos,
                mesh_shape=mesh_shape,
                paint_absolute_pos=paint_absolute_pos,
                 halo_size=halo_size,
                 sharding=sharding,
            )* 1.5* cosmo.Omega_m
        ain = jnp.atleast_1d(a)
        ain = jnp.abs(ain)
        forces = lax.cond(args[-2]==1, forcefun, lambda _: forces, None)#        forces = forcefun(None)#
        dvel = 1.0 / (a**2 * E(cosmo, a)) * forces
        return dvel, forces
    def __inner(a, x, args):
        if args[-1]==0: #Kick
            k, f= kick(a, x[0], args, x[2])
            f = lax.cond(args[-2]==1, lambda _: f, lambda _: jnp.zeros_like(f), None)
            return jnp.stack([jnp.zeros_like(x[1]),k,f], axis=0)
        if args[-1]==1: #Drift 
            return jnp.stack([drift(a, x[1], args),jnp.zeros_like(x[0]), jnp.zeros_like(x[2])], axis=0)
    return __inner

def drift_factor_in(a_c, a_prev, a_next, cosmo):
    """
    fastpm eq 24
    """
    D_next = growth_factor(cosmo, a_next)
    D_prev = growth_factor( cosmo, a_prev)
    g_D = growth_rate(cosmo, a_c)/a_c*growth_factor(cosmo, a_c) 
    return (D_next - D_prev) / g_D

def kick_factor_in(a_c, a_prev, a_next, cosmo):
    """
    fastpm eq 25 
    """
    return (Gf(cosmo,a_next)-Gf(cosmo,a_prev))/dGfa(cosmo,a_c)



