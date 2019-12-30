from HolodeckEnv import HolodeckEnv

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel


def build_and_train(name, run_ID=0, cuda_idx=0):
    sampler = SerialSampler(
        EnvCls=HolodeckEnv,
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        env_kwargs=dict(scenario_name="MazeWorld-MazeRlpyt"),
        eval_env_kwargs=dict(scenario_name="MazeWorld-MazeRlpyt"),
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = DQN(min_steps_learn=1e3)  # Run with defaults.
    agent = DqnAgent(
        ModelCls=AtariDqnModel,  # TODO rewrite model class and params
        model_kwargs={
            "image_shape": (4, 32, 32),
            "output_size": 4,
        }
    )
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    log_dir = name
    with logger_context(log_dir, run_ID, name, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    build_and_train(
        "dqn_holodeck_test"
    )