"""
This script is used to train IQL agents using the TorchRL library in a traffic simulation environment.
The IQL implementation is based on: https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/iql.py
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
import argparse
import ast
import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch

from routerl import TrafficEnvironment
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives.value import GAE
from torchrl.objectives import SoftUpdate, ValueEstimators, DQNLoss
from tqdm import tqdm

from utils import clear_SUMO_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    args = parser.parse_args()
    ALGORITHM = "iql_torchrl"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"PyTorch seed: {torch_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device is: ", device)

     
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], alg_params, env_params, task_params


    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    
    custom_network_folder = f"../networks/{network}"
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"

    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    
    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
    num_machines = int(num_agents * ratio_machines)
    training_episodes = agent_frames_per_batch * n_iters
    frames_per_batch = num_machines * agent_frames_per_batch
    total_frames = frames_per_batch * n_iters
    phases = [1, human_learning_episodes, int(training_episodes) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]  
    
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["algorithm"] = ALGORITHM
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    # Initiate the traffic environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" :{
                "behavior" : av_behavior,
            }
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
        },  
        environment_parameters = {
            "save_every" : save_every,
        },
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "visualize_paths" : False
        } 
    )
    
    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)
    
    env.start()
    env.reset()

     
    # Human learning
    pbar = tqdm(total=human_learning_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()
    pbar.close()
     
    #  Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    
    print(f"""
    Agents in the traffic:
    • Total agents           : {len(env.all_agents)}
    • Human agents           : {len(env.human_agents)}
    • AV agents              : {len(env.machine_agents)}
    """)
    
    
    group = {'agents': [str(machine.id) for machine in env.machine_agents]}

    env = PettingZooWrapper(
        env=env,
        use_mask=True,
        categorical_actions=True,
        done_on_any = False,
        group_map=group,
        device=device
    )

     
    # Transforms
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    
    check_env_specs(env)
    env.reset()

     
    #  Policy network
    # > Instantiate an `MPL` that can be used in multi-agent contexts.
    
    net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=env.action_spec.space.n,
            n_agents=env.n_agents,
            centralised=False,
            share_params=False,
            device=device,
            depth=mlp_depth,
            num_cells=mlp_cells,
            activation_class=nn.ReLU,
        )

    
    module = TensorDictModule(
            net, in_keys=[("agents", "observation")], out_keys=[("agents", "action_value")]
    )

    value_module = QValueModule(
        action_value_key=("agents", "action_value"),
        out_keys=[
            env.action_key,
            ("agents", "action_value"),
            ("agents", "chosen_action_value"),
        ],
        spec=env.action_spec,
        action_space=None,
    )

    qnet = SafeSequential(module, value_module)

    
    qnet_explore = TensorDictSequential(
        qnet,
        EGreedyModule(
            eps_init=eps_init,
            eps_end=eps_end,
            annealing_num_steps=int(total_frames * exploration_fraction),
            action_key=env.action_key,
            spec=env.action_spec,
        ),
    )

     
    # Collector
    collector = SyncDataCollector(
            env,
            qnet_explore,
            device=device,
            storing_device=device,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
        )

     
    # Replay buffer
    replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(memory_size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=minibatch_size,
        )

     
    # DQN loss function
    loss_module = DQNLoss(qnet, delay_value=True)

    loss_module.set_keys(
            action_value=("agents", "action_value"),
            action=env.action_key,
            value=("agents", "chosen_action_value"),
            reward=env.reward_key,
            done=("agents", "done"),
            terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    target_net_updater = SoftUpdate(loss_module, eps=eps)

    optim = torch.optim.Adam(loss_module.parameters(), lr)

     
    # Training loop
    loss_values_path = os.path.join(records_folder, "losses/loss_values.txt")
    os.makedirs(os.path.dirname(loss_values_path), exist_ok=True)
    open(loss_values_path, 'w').close()
    
    pbar = tqdm(total=n_iters, desc="Training")
    for tensordict_data in collector:
        current_frames = tensordict_data.numel()
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)
        ## Update the policies of the learning agents
        step_loss_values = []
        for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = loss_vals["loss"]
                step_loss_values.append(loss_value.item())
                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)

                optim.step()
                optim.zero_grad()
            target_net_updater.step()

        if step_loss_values:
            loss = sum(step_loss_values) / len(step_loss_values)
            with open(loss_values_path, 'a') as f:
                f.write("%s\n" % loss)
        qnet_explore[1].step(frames=current_frames)  # Update exploration annealing
        collector.update_policy_weights_()
        pbar.update()
    
    pbar.close()
    collector.shutdown()
    
    # Testing phase
    pbar = tqdm(total=test_eps, desc="Test phase")
    qnet.eval() # set the policy into evaluation mode
    for episode in range(test_eps):
        env.rollout(len(env.machine_agents), policy=qnet)
        pbar.update()
    pbar.close()
        
    os.makedirs(plots_folder, exist_ok=True)
    env.plot_results()
    
    # Visualize losses
    loss_values = list()
    with open(loss_values_path, 'r') as f:
        for line in f:
            loss_values.append(float(line.strip()))    
    colors = [
        "firebrick", "teal", "peru", "navy", 
        "salmon", "slategray", "darkviolet", 
        "lightskyblue", "darkolivegreen", "black"]
    plt.figure(figsize=(12, 6))
    plt.plot(loss_values, label='loss_values', color=colors[0], linewidth=3)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Loss', fontsize=18, fontweight='bold')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'losses.png'), dpi=300)
    plt.close()
    
    env.stop_simulation()

    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)


