from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime
import re

class Logger:
  def __init__(self, config):
    self.config = config
    self.log_file_path = self.setup_logging_folder()

    self.writer = SummaryWriter(self.log_file_path)
    self.write_config_to_file()

    self.score = []
    self.loss = []
    self.average_score = []

    print("Logging at: {}".format(self.log_file_path))
  
  def log_scalar(self, name, value, episode):
    self.writer.add_scalar(name, value, episode)

    if (name == "score"):
      self.score.append(value)
    if (name == "loss"):
      self.loss.append(value)
    if (name == "average_score"):
      self.average_score.append(value)

  def setup_logging_folder(self):
    # Get env name
    result = re.search('EnvSpec((.*))', str(self.config.env.spec))
    env_name = result.group(1)[1:-1]

    if self.config.save_loc:
      log_file_path = "logs/" + env_name +"/{}-{date:%Y-%m-%d_%H_%M_%S}".format(self.config.save_loc, date=datetime.datetime.now())
    else:
      log_file_path = "logs/" + env_name +"/{}-{date:%Y-%m-%d_%H_%M_%S}".format("experiment", date=datetime.datetime.now())
    Path(log_file_path).mkdir(parents=True, exist_ok=True)

    return log_file_path

  def write_config_to_file(self):
    file = open(self.log_file_path + "/configuration.txt","w") 
 
    file.write("env: {}\n".format(self.config.env.spec))
    file.write("win condition: {}\n".format(self.config.win_condition))
    file.write("state space{}\n".format(self.config.env.observation_space.shape))
    file.write("action space{}\n".format(self.config.env.action_space.n))


    file.write("device: {}\n".format(self.config.device))
    file.write("seed: {}\n".format(self.config.seed))

    file.write("n_episodes: {}\n".format(self.config.n_episodes))
    file.write("max_t: {}\n".format(self.config.max_t))
    file.write("eps_start: {}\n".format(self.config.eps_start))
    file.write("eps_end: {}\n".format(self.config.eps_end))
    file.write("eps_decay: {}\n".format(self.config.eps_decay))

    file.write("eps_greedy: {}\n".format(self.config.eps_greedy))
    file.write("noisy: {}\n".format(self.config.noisy))


    file.write("tau: {}\n".format(self.config.tau))
    file.write("gamma: {}\n".format(self.config.gamma))
    file.write("lr: {}\n".format(self.config.lr))

    file.write("memory: {}\n".format(self.config.memory))
    file.write("batch_size: {}\n".format(self.config.batch_size))
    file.write("buffer_size: {}\n".format(self.config.buffer_size))
    file.write("lr_annealing: {}\n".format(self.config.lr_annealing))
    file.write("learn_every: {}\n".format(self.config.learn_every))

    file.write("double_dqn: {}\n".format(self.config.double_dqn))
    file.write("model: {}\n".format(self.config.model))

    file.write("save_loc: {}\n".format(self.config.save_loc))
    
    file.close() 