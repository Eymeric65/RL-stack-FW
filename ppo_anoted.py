# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# QU'est ce que je cherche a comprendre ? 

# A quoi sert le reseau de critique et qu'est-ce qu'il prevoit

# sur quel loss se base la regression




@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module): # class de l'agent contient un reseau de critique et un reseau de decision
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ) # le reseau de critique est unitaire, il cherche a prevoir la reward (ou le loss... a voir)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        ) # le reseau acteur est le reseau qui prevoit les actions future.

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x) # Calcul des actions grace au reseau de l'acteur
        probs = Categorical(logits=logits) # Convertis l'output en tableau de proba
        if action is None:
            action = probs.sample() # Choisis l'action dans le champs des proba
        return action, probs.log_prob(action), probs.entropy(), self.critic(x) # l'action, le logarithme de la probabilite choisis, l'entropie de la distribution, critique de la reward


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps) # quel est la taille d'un batch
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # Quel est la taille du sous batch qui fait le calcul d'optimisation ( une epoch c'est num_minibatches optimisation)
    args.num_iterations = args.total_timesteps // args.batch_size # Nombre de sample 
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track: # Implementation de Weight and Biases
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}") # Writer pour Tensorboard 
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed) # Seending...
    np.random.seed(args.seed) # Seending...
    torch.manual_seed(args.seed) # Seending...
    torch.backends.cudnn.deterministic = args.torch_deterministic # Seending...

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu") # Selection du GPU

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    ) # Creation des environment (vectorise ou pas)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device) # creation du module de reseaux de neuronnes en fonction de l'environment et envoie du reseaux de neuronnes sur le GPU
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5) # Initialisation de l'optimisateur de descente de gradient

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device) # Tableau des observation sur le batch
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device) # Tableau des actions
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device) # Tableau de l'entropie ?
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device) # Tableau des reward 
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device) # Tableau si complete
    values = torch.zeros((args.num_steps, args.num_envs)).to(device) # HMM ?

    # TRY NOT TO MODIFY: start the game
    global_step = 0 # Compteur globaux
    start_time = time.time() # Temps au debut
    next_obs, _ = envs.reset(seed=args.seed) # Initialisation des environement
    next_obs = torch.Tensor(next_obs).to(device) # Envoie du tableau sur le GPU
    next_done = torch.zeros(args.num_envs).to(device) # Creation et envoie du tableau des evenement done

    for iteration in range(1, args.num_iterations + 1): # Nombre de fois que l'on fait le training ( L 141 )
        # Annealing the rate if instructed to do so.
        if args.anneal_lr: # est ce que l'on reduit le learning rate durant les differentes iteration (de base oui)
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow # Diminution jusqu'a 0

        for step in range(0, args.num_steps): # Combien de pas par update du reseau de neuronnes
            global_step += args.num_envs # increment du nombre de pas globaux
            obs[step] = next_obs # on sauvegarde les anciennes observation dans le logger
            dones[step] = next_done # exactement pareil

            # ALGO LOGIC: action logic
            with torch.no_grad(): # c'est un step de simulation donc on ne veut pas stocker les gradient d'execution
                action, logprob, _, value = agent.get_action_and_value(next_obs) # On obtient les actions predites par le modele
                values[step] = value.flatten() # la critique
            actions[step] = action # store de l'action
            logprobs[step] = logprob # store le logarithme de la probabilite de l'action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy()) # Mise des actions sur le CPU et avancement des environment de simulation
            next_done = np.logical_or(terminations, truncations) # Si il y a eu terminaison ou tronquaison
            rewards[step] = torch.tensor(reward).to(device).view(-1) # mise du reward dans le gpu 
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device) # mise des observations et de la done dans le gpu

            if "final_info" in infos: # log si il y a des infos ( a etudier )
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done # Je comprends pas trop...
        with torch.no_grad(): # Step de simulation ?  Action final
            next_value = agent.get_value(next_obs).reshape(1, -1) # get critic 
            advantages = torch.zeros_like(rewards).to(device) # Initialisation d'un array de la taille du batch
            lastgaelam = 0
            for t in reversed(range(args.num_steps)): # Debut par la fin 
                if t == args.num_steps - 1: # Exception du step de fin (car action final)
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                # Recherche du GAE lambda ... manque theorique
                
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
