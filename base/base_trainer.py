import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils import filter_state_dict

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, config):
        self.config = config

        # 根据trainer或者evaluater设置参数
        if "trainer" in config.config:
            self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
            cfg_trainer = config['trainer']
            self.epochs = cfg_trainer['epochs']
            self.save_period = cfg_trainer['save_period']
            self.monitor = cfg_trainer.get('monitor', 'off')

        else:
            self.logger = config.get_logger("evaluater")
            self.monitor = "off"

        # setup GPU device if available, move model into configured device
        # 设置GPU
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        # 设置其他参数
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.save_multiple = True

        # configuration to monitor model performance and save best
        # 监测模型表现，并存最佳参数
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        # 设置可视化参数
        if "trainer" in config.config:
            self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    # 带有抽象方法的类是抽象类，不能被实例化，继承了含抽象方法的子类必须复写抽象方法
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        # epoch循环
        for epoch in range(self.start_epoch, self.epochs + 1):
            # 训练一个epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            # 保存log信息
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            # 向屏幕中打印log信息
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            # 根据配置的指标评估模型性能，将最佳检查点保存为 model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    # 根据指定的度量（mnt_metric）检查模型性能是否有所提高
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            # 固定每个周期保存checkpoint
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        如果可行的化，设置 GPU 设备（如果可用），将模型移动到配置的设备中
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number 当前的epoch
        :param log: logging information of the epoch 当前epoch的log信息
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth' 是否保存为model_best.pth
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if self.save_multiple:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        else:
            filename = str(self.checkpoint_dir / 'checkpoint.pth')
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        从保存的检查点恢复
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        # self.model.load_state_dict(checkpoint['state_dict'])
        checkpoint_state_dict = filter_state_dict(checkpoint["state_dict"], checkpoint["arch"] == "DataParallel" and len(self.device_ids) == 1)
        self.model.load_state_dict(checkpoint_state_dict, strict=False)

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
