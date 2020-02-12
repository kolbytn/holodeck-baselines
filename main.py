import os
from absl import app, flags

from holodeck_env import HolodeckEnv
from holodeck_model import PpoHolodeckModel

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.agents.pg.gaussian import GaussianPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context


FLAGS = flags.FLAGS

flags.DEFINE_integer('run_id', 0, 'Run ID of experiment.')
flags.DEFINE_integer('cuda_idx', 0, 'Index of cuda device.')
flags.DEFINE_integer('n_steps', int(1e6), 'Experiment length.')
flags.DEFINE_integer('log_steps', int(1e4), 'Log frequency.')
flags.DEFINE_integer('eps_length', int(200), 'Episode Length.')
flags.DEFINE_integer('sampler_steps', int(80), 'Steps in sampler itr.')
flags.DEFINE_integer('num_workers', int(1), 'Num parellel workers.')
flags.DEFINE_integer('gif_freq', int(500), 'How often to create gifs.')
flags.DEFINE_integer('hidden_size', int(1024), 'Model hidden size.')
flags.DEFINE_string('name', 'test', 'Name of experiment.')
flags.DEFINE_string('checkpoint', None, 'Path to model checkpoint')
flags.DEFINE_string('image_dir', 'images', 'Path to saved gifs')
flags.DEFINE_string('scenario', 'InfiniteForest-MaxDistance', 'Scenario to use')


def train_holodeck_ppo(argv):

    # Create gif directory
    image_path = os.path.join(FLAGS.image_dir, FLAGS.name)
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))

    # Load saved checkpoint
    if FLAGS.checkpoint is not None:
        checkpoint = torch.load(FLAGS.checkpoint)
        model_state_dict = checkpoint['agent_state_dict']
        optim_state_dict = checkpoint['optim_state_dict']
    else:
        model_state_dict = None
        optim_state_dict = None

    # Get environment info for agent
    env = HolodeckEnv(scenario_name=FLAGS.scenario, 
                      max_steps=FLAGS.eps_length, 
                      gif_freq=FLAGS.gif_freq, 
                      image_dir=image_path)

    # Instantiate sampler
    sampler = SerialSampler(
        EnvCls=HolodeckEnv,
        batch_T=FLAGS.sampler_steps,
        batch_B=FLAGS.num_workers,
        env_kwargs=dict(scenario_name=FLAGS.scenario, 
                        max_steps=FLAGS.eps_length, 
                        gif_freq=FLAGS.gif_freq, 
                        image_dir=image_path),
        max_decorrelation_steps=0,
    )

    # Instantiate algo and agent
    algo = PPO(initial_optim_state_dict=optim_state_dict)

    AgentClass = GaussianPgAgent \
        if env.is_action_continuous \
            else CategoricalPgAgent
    agent = AgentClass(
        initial_model_state_dict=model_state_dict,
        ModelCls=PpoHolodeckModel,
        model_kwargs={
            'img_size': env.img_size,
            'lin_size': env.lin_size, 
            'action_size': env.action_size,
            'is_continuous': env.is_action_continuous,
            'hidden_size': FLAGS.hidden_size
        }
    )

    # Instantiate runner
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=FLAGS.n_steps,
        log_interval_steps=FLAGS.log_steps,
        affinity=dict(cuda_idx=FLAGS.cuda_idx), 
                      #workers_cpus=list(range(FLAGS.num_workers)))
    )

    # Run
    params = {
        'run_id': FLAGS.run_id,
        'cuda_idx': FLAGS.cuda_idx,
        'n_steps': FLAGS.n_steps,
        'log_steps': FLAGS.log_steps,
        'eps_length': FLAGS.eps_length,
        'sampler_steps': FLAGS.sampler_steps,
        'num_workers': FLAGS.num_workers,
        'gif_freq': FLAGS.gif_freq,
        'hidden_size': FLAGS.hidden_size,
        'name': FLAGS.name,
        'checkpoint': FLAGS.checkpoint,
        'image_dir': FLAGS.image_dir,
        'scenario': FLAGS.scenario
    }
    with logger_context(FLAGS.name, FLAGS.run_id, FLAGS.name, 
                        snapshot_mode='all', log_params=params):
        runner.train()


if __name__ == "__main__":
    app.run(train_holodeck_ppo)
