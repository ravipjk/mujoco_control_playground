"""SAC with a Huber (robust) critic loss instead of MSE.

Why this exists
---------------
On the 153-dim CPG action space the SAC critic over-estimates and DIVERGES:
Q inflates to 1e4-1e6 while the true |Q| is ~1e2. Lowering the learning rate
(3e-4 -> 1e-4 -> 3e-5) only slowed the blow-up ~10-30x per step without ever
crossing into a bounded regime -- an "LR floor": the rate rescales the loop but
does not change the sign of the feedback.

The mechanism is a positive-feedback loop: an over-estimated bootstrap target
Q' inflates y = r + gamma*(min Q' - alpha*logpi'); the critic regresses toward
the inflated y; that raises the next target; repeat. With MSE the per-sample
gradient is `r` (the TD residual) -- UNBOUNDED -- so as residuals grow the loop
GAIN grows with them, giving super-linear blow-up.

Huber caps it. huber_loss(delta) is quadratic (== MSE) for |r| <= delta and
LINEAR beyond, so its gradient SATURATES at +/-delta. Once residuals exceed
delta the critic still creeps toward a bad target but at a bounded rate -- the
explosive amplifier becomes a bounded integrator. (Same idea as an actuator
saturation against wind-up, or a robust M-estimator treating inflated targets as
outliers -- which is exactly why DQN/Rainbow use it for value learning.)

We use torch.huber_loss (NOT smooth_l1_loss): huber keeps the small-residual
region as 0.5*r^2, identical to SB3's `0.5 * mse_loss`, so below the knee this is
a clean A/B against plain SAC and only the tail behaviour changes. smooth_l1
divides the quadratic region by beta, which would rescale the effective critic LR.

Only `train()` is overridden; it is copied verbatim from SB3 2.8.0
(stable_baselines3/sac/sac.py) with the single critic-loss line changed.
"""
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class HuberSAC(SAC):
    """SAC whose critic regresses under a Huber loss with knee ``huber_delta``.

    ``huber_delta`` should sit ABOVE the healthy TD-residual RMS (so normal value
    fitting stays MSE-like) and BELOW the divergent residual scale (so the runaway
    gets clipped). For this env at gamma=0.99 the healthy RMS is ~25 and divergence
    reaches 100s-1000s, so ~50 is a reasonable default. Raise it if you raise gamma
    (larger returns -> larger residuals).
    """

    def __init__(self, *args, huber_delta: float = 50.0, **kwargs):
        # set before super().__init__ so it exists no matter what init touches
        self.huber_delta = float(huber_delta)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss -- HUBER (robust) instead of SB3's MSE. huber_loss is
            # 0.5*r^2 for |r|<=delta (== SB3's `0.5 * mse_loss`) and delta*(|r|-0.5*delta)
            # beyond, so the per-sample gradient saturates at +/-delta and the bootstrap
            # positive-feedback loop can no longer self-amplify. (See module docstring.)
            critic_loss = sum(
                F.huber_loss(current_q, target_q_values, delta=self.huber_delta)
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
