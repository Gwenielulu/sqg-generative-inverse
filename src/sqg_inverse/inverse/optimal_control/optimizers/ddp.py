import torch
import numpy as np
import time
from .util import get_Q, get_jmp_fn, get_mjp_fn, inv
from tqdm import tqdm
import gc


class DDP:

    def __init__(
        self,
        env,
        k_jacobian=1,
        k_hessian=1,
        lr_mode="k",
        rrf_iters=2,
        eps=0.001,
        success_multiplier=1.0,
        failure_multiplier=10.0,
        min_eps=1e-08,
        chunk_size=8,
        verbose=1,
        print_every=10,
        use_running_state_cost=False,
        lr_identity=False,
        seed=None,
    ):
        self.env = env
        self.k_jacobian = k_jacobian
        self.k_hessian = k_hessian
        self.lr_mode = lr_mode
        self.eps = eps
        self.success_multiplier = success_multiplier
        self.failure_multiplier = failure_multiplier
        self.min_eps = min_eps
        self.chunk_size = chunk_size
        self.rrf_iters = rrf_iters
        self.verbose = verbose
        self.print_every = print_every
        self.use_running_state_cost = use_running_state_cost
        self.lr_identity = lr_identity
        self.seed = seed

    def solve(self, init_state, actions=None, num_iterations=25, early_stopping=True):
        actions_shape = (self.env.num_steps - 1, len(init_state), self.env.control_dim)
        if actions is None:
            actions = torch.zeros(size=actions_shape, device=init_state.device)
        else:
            actions = actions.reshape(actions_shape)
        states = None
        end_states = []
        time0 = time.time()
        self.pbar = tqdm(range(num_iterations))
        for iter_num in self.pbar:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            gc.collect()
            torch.cuda.empty_cache()
            states, actions = self.iterate(
                init_state.clone(),
                actions,
                iter_num=iter_num,
                states=states,
                last=iter_num == num_iterations - 1,
            )
            end_states.append(states[-1].cpu())
        return (actions.squeeze(), end_states)

    def get_cost(self, states, actions):
        running_cost = 0
        for t, state, action in zip(self.env.timesteps[:-1], states[:-1], actions):
            state, action = (state.reshape(1, -1), action.reshape(1, -1))
            running_cost += self.env.running_control_cost(
                t, x=state, u=action
            ).squeeze()
            if self.use_running_state_cost:
                running_cost += self.env.running_state_cost(t, x=state, u=action)
        terminal_cost = self.env.terminal_cost(states[-1].reshape(1, -1))
        return (running_cost, terminal_cost)

    def update_actions(
        self, states, actions, ks, Ks, iter_num, num_tries=20, last=False
    ):
        assert len(state) == 1
        alpha = 1.0
        n_steps, _ = actions.shape
        running_cost, terminal_cost = self.get_cost(states, actions)
        orig_cost = running_cost + terminal_cost
        for _ in range(num_tries):
            state = states[:1]
            cand_actions = []
            new_states = [state]
            for t, orig_state, action, k, K in zip(
                self.env.timesteps[:-1], states, actions, ks, Ks
            ):
                res = state - orig_state
                new_action = action.unsqueeze(-1) + alpha * k + K @ res.unsqueeze(-1)
                new_action = new_action.reshape(1, self.env.control_dim)
                state = self.env.step(t, state, new_action)
                new_states.append(state)
                cand_actions.append(new_action)
            cand_actions = torch.cat(cand_actions)
            running_cost, terminal_cost = self.get_cost(new_states, cand_actions)
            cand_cost = running_cost + terminal_cost
            if cand_cost < orig_cost:
                self.eps *= self.success_multiplier
                break
            alpha *= 0.5
        else:
            self.eps *= self.failure_multiplier
            print(f"linesearch failed, eps={self.eps:.3e}")
            cand_cost = orig_cost
            cand_actions = actions
        return (new_states, cand_actions)

    def running_cost_derivatives(self, states, actions, projected=False):
        n_steps, d_b, _ = states.shape
        d_s, d_c = (self.env.state_dim, self.env.control_dim)
        states = states.reshape(n_steps * d_b, d_s)
        actions = actions.reshape(n_steps * d_b, d_c)
        if not self.use_running_state_cost:
            batched_timesteps = self.env.timesteps[:-1].tile((d_b,))
            lx, lu = self.env.j_cost(batched_timesteps, states, actions)
            (lxx, lxu), (lux, luu) = self.env.h_cost(
                batched_timesteps, states, actions, low_mem_mode=projected
            )
            q = torch.ones((len(states), d_s, 1), device=states.device)
            qTlxx = torch.zeros_like(q.mT)
        else:
            _, lu = self.env.j_cost(self.env.timesteps[:-1], states, actions)
            running_state_cost = lambda x: self.env.running_state_cost(
                self.env.timesteps[:-1], x, actions
            ).sum()
            lx_fn = torch.func.jacrev(running_state_cost)
            lx = lx_fn(states)
            (_, lxu), (lux, luu) = self.env.h_cost(
                self.env.timesteps[:-1], states, actions, low_mem_mode=projected
            )
            (q,), (qTlxx,) = self.lr_jacobian(
                lx_fn, states, k=self.k_hessian, mode=self.lr_mode
            )
            qTlxx = qTlxx @ q @ q.mT
        lx = lx.reshape(n_steps, d_b, d_s, 1)
        lu = lu.reshape(n_steps, d_b, d_c, 1)
        if not projected:
            lxx = q @ qTlxx
            lxx = lxx.reshape(n_steps, d_s, d_s)
            lxu = lxu.reshape(n_steps, d_s, d_c)
            lux = lux.reshape(n_steps, d_c, d_s)
            luu = luu.reshape(n_steps, d_c, d_c)
            return (lx, lu, lxx, luu, lxu, lux)
        return (q, lx, lu, qTlxx, luu, lxu, lux)

    def lr_jacobian(self, fn, *xs, k=1, mode="2k", qs=None):
        if mode == "k":
            output, mjp_fn = get_mjp_fn(
                fn, *xs, return_output=True, chunk_size=self.chunk_size
            )
            qs, qTJs = self.low_rank_approx(
                mjp_fn,
                output_shapes=(output.shape,),
                input_shapes=[x.shape for x in xs],
                k=k,
                mode=mode,
                qs=qs,
                device=xs[0].device,
            )
        elif mode == "2k":
            A = get_mjp_fn(fn, *xs, chunk_size=self.chunk_size)
            AT = get_jmp_fn(fn, *xs, chunk_size=self.chunk_size)
            qs, qTJs = self.low_rank_approx(
                A,
                AT=AT,
                input_shapes=[x.shape for x in xs],
                k=k,
                mode=mode,
                qs=qs,
                device=xs[0].device,
            )
        return (qs, qTJs)

    def low_rank_approx(
        self,
        A,
        input_shapes=None,
        output_shapes=None,
        AT=None,
        k=1,
        mode="2k",
        qs=None,
        device="cpu",
    ):
        if mode == "k":
            if qs is None:
                qs = tuple((get_Q(shape, k, device=device) for shape in output_shapes))
            if k > 0:
                qTAs = A(*qs)
            else:
                qs = tuple((get_Q(shape, 1, device=device) for shape in output_shapes))
                b, d_out = output_shapes[0]
                qTAs = tuple(
                    [
                        torch.zeros(size=(b, 1, d_out), device=device)
                        for _ in range(len(input_shapes))
                    ]
                )
        elif mode == "2k":
            if qs is None:
                qs = tuple((get_Q(shape, k, device=device) for shape in input_shapes))
            for _ in range(self.rrf_iters - 1):
                qs = AT(qs)
                qTAs = A(qs)
                assert isinstance(qTAs, tuple)
                qs = tuple((q.mT for q in qTAs))
            qs = AT(qs)
            if isinstance(qs, tuple):
                qs = tuple((torch.linalg.qr(q)[0] for q in qs))
                qTAs = A(qs)
            else:
                qs = torch.linalg.qr(qs)[0]
                qTAs = A(qs)
                qs = (qs,)
        return (qs, qTAs)

    def compute_gradients(self, states, actions):
        d_s, d_c = (self.env.state_dim, self.env.control_dim)
        n_steps, _ = actions.shape
        state = states[-1]
        Vx_fn = torch.func.jacrev(self.env.terminal_cost)
        Vx = Vx_fn(state.reshape(1, d_s))
        (qv,), (qTVxx,) = self.lr_jacobian(
            Vx_fn, state.reshape(1, d_s), k=self.k_hessian, mode=self.lr_mode
        )
        qv, qTVxx = (qv[0], qTVxx[0])
        Vxx_asym = qv @ qTVxx
        Vxx = Vxx_asym @ qv @ qv.mT
        if self.lr_identity:
            I = torch.eye(d_s, device=state.device)
            fn = lambda x, u: self.env.step(self.env.timesteps[:-1], x, u) - (x + u)
        else:
            fn = lambda x, u: self.env.step(self.env.timesteps[:-1], x, u)
        (qs,), (qTfxs, qTfus) = self.lr_jacobian(
            fn, states[:-1], actions, k=self.k_jacobian, mode=self.lr_mode
        )
        lxs, lus, lxxs, luus, lxus, luxs = self.running_cost_derivatives(
            states[:-1], actions
        )
        ks = [None] * n_steps
        Ks = [None] * n_steps
        for t in range(n_steps - 1, -1, -1):
            state, action = (states[t].clone(), actions[t].clone())
            lx, lu = (lxs[t], lus[t])
            lxx, luu, lxu, lux = (lxxs[t], luus[t], lxus[t], luxs[t])
            q, qTfx, qTfu = (qs[t], qTfxs[t], qTfus[t])
            if self.lr_identity:
                fx, fu = (q @ qTfx + I, q @ qTfu + I)
            else:
                fx, fu = (q @ qTfx, q @ qTfu)
            Qxx = lxx + fx.T @ Vxx @ fx
            Quu = luu + fu.T @ Vxx @ fu
            Qux = lux + fu.T @ Vxx @ fx
            Qxu = lxu + fx.T @ Vxx @ fu
            step_fn = lambda x, u: self.env.step(self.env.timesteps[t], x, u)
            fxT_Vx, fuT_Vx = torch.func.vjp(step_fn, state, action)[1](
                Vx.reshape(1, d_s)
            )
            Qx = lx + fxT_Vx.reshape(d_s, 1)
            Qu = lu + fuT_Vx.reshape(d_c, 1)
            Quu_inv = inv(Quu, self.eps)
            k = -Quu_inv @ Qu
            K = -0.5 * Quu_inv @ (Qxu.mT + Qux)
            Vx = Qx - K.mT @ Quu @ k
            Vxx = Qxx - K.mT @ Quu @ K
            ks[t] = k.detach()
            Ks[t] = K.detach()
        ks = torch.stack(ks)
        Ks = torch.stack(Ks)
        return (ks, Ks)

    def iterate(self, init_state, actions, iter_num, states=None, last=False):
        with torch.no_grad():
            self.eps = np.clip(self.eps, a_min=self.min_eps, a_max=np.inf)
            if states is None:
                states = [init_state]
                for t, action in zip(self.env.timesteps[:-1], actions):
                    state = self.env.step(t, states[-1], action)
                    states.append(state)
                states = torch.stack(states, axis=0)
            gradients = self.compute_gradients(states, actions)
            states, actions = self.update_actions(
                states, actions, *gradients, iter_num=iter_num, last=last
            )
        return (states.detach(), actions.detach())
