import logging
from typing import Dict, Optional, Tuple, Union

from calvin_agent.models.calvin_base_model import CalvinBaseModel
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import torch

from taowm.models.decoders.action_decoder import ActionDecoder

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class GCBC(pl.LightningModule, CalvinBaseModel):
    """
    The lightning module used for training.

    Args:
        perceptual_encoder: DictConfig for perceptual_encoder.
        visual_goal: DictConfig for visual_goal encoder.
        action_decoder: DictConfig for action_decoder.
        state_recons: If True, use state reconstruction auxiliary loss.
        state_recon_beta: Weight for state reconstruction loss term.
        optimizer: DictConfig for optimizer.
        lr_scheduler: DictConfig for learning rate scheduler.
    """

    def __init__(
        self,
        perceptual_encoder: DictConfig,
        visual_goal: DictConfig,
        action_decoder: DictConfig,
        state_recons: bool,
        state_recon_beta: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
    ):
        super(GCBC, self).__init__()
        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder, device=self.device)
        self.setup_input_sizes(
            self.perceptual_encoder,
            visual_goal,
            action_decoder,
        )

        # goal encoders
        self.visual_goal = hydra.utils.instantiate(visual_goal)

        # policy network
        self.action_decoder: ActionDecoder = hydra.utils.instantiate(action_decoder)

        # auxiliary losses
        self.state_recons = state_recons
        self.st_recon_beta = state_recon_beta

        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler

        self.save_hyperparameters()

        # for inference
        self.rollout_step_counter = 0
        self.latent_goal = None

    @staticmethod
    def setup_input_sizes(
        perceptual_encoder,
        visual_goal,
        action_decoder,
    ):
        """
        Configure the input feature sizes of the respective parts of the network.

        Args:
            perceptual_encoder: DictConfig for perceptual encoder.
            visual_goal: DictConfig for visual goal encoder.
            action_decoder: DictConfig for action decoder network.
        """
        visual_goal.in_features = perceptual_encoder.latent_size
        action_decoder.perceptual_features = perceptual_encoder.latent_size
        action_decoder.plan_features = 0  # No planning for GCBC

    @property
    def num_training_steps(self) -> int:
        """
        Total training steps inferred from datamodule and devices.

        Returns:
            Number of estimated training steps.
        """
        assert isinstance(self.trainer, pl.Trainer)
        combined_loader_dict = self.trainer.datamodule.train_dataloader()  # type: ignore
        dataset_lengths = [len(combined_loader_dict[k]) for k in combined_loader_dict.keys()]
        dataset_size = max(dataset_lengths)
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices  # type: ignore
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs  # type: ignore

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:  # type: ignore
            return self.trainer.max_steps  # type: ignore
        return max_estimated_steps

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        """
        Set up warmup steps for learning rate scheduler.

        Args:
            num_training_steps: Number of training steps, if < 0 infer from class attribute.
            num_warmup_steps: Either as absolute number of steps or as percentage of training steps.

        Returns:
            num_training_steps: Number of training steps for learning rate scheduler.
            num_warmup_steps: Number of warmup steps for learning rate scheduler.
        """
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        num_warmup_steps = int(num_warmup_steps)
        return num_training_steps, num_warmup_steps

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        if "num_warmup_steps" in self.lr_scheduler:
            self.lr_scheduler.num_training_steps, self.lr_scheduler.num_warmup_steps = self.compute_warmup(
                num_training_steps=self.lr_scheduler.num_training_steps,
                num_warmup_steps=self.lr_scheduler.num_warmup_steps,
            )
            rank_zero_info(f"Inferring number of training steps, set to {self.lr_scheduler.num_training_steps}")
            rank_zero_info(f"Inferring number of warmup steps from ratio, set to {self.lr_scheduler.num_warmup_steps}")
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.


        Returns:
            loss tensor
        """
        action_loss, proprio_loss, total_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        batch_size: Dict[str, int] = {}
        total_bs = 0
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            latent_goal = self.visual_goal(perceptual_emb[:, -1])
            if self.state_recons:  # skip
                proprio_loss += self.perceptual_encoder.state_reconstruction_loss()

            robot_obs = dataset_batch["state_info"]["robot_obs"]
            actions = dataset_batch["actions"]
            empty_plan = torch.empty((dataset_batch["actions"].shape[0]), 0).to(self.device)
            act_loss = self.action_decoder.loss(empty_plan, perceptual_emb, latent_goal, actions, robot_obs)

            action_loss += act_loss
            total_loss += act_loss
            batch_size[self.modality_scope] = dataset_batch["actions"].shape[0]
            total_bs += dataset_batch["actions"].shape[0]

            self.log(
                f"train/action_loss_{self.modality_scope}",
                act_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
                sync_dist=True,
            )
        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
        action_loss = action_loss / len(batch)
        if self.state_recons:
            proprio_loss = proprio_loss / len(batch)
            total_loss = total_loss + self.st_recon_beta * proprio_loss
            self.log(
                "train/pred_proprio",
                self.st_recon_beta * proprio_loss,
                on_step=False,
                on_epoch=True,
                batch_size=total_bs,
                sync_dist=True,
            )
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, batch_size=total_bs, sync_dist=True)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, batch_size=total_bs, sync_dist=True)
        return total_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal networks, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss = torch.tensor(0.0).to(self.device)
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"]
            )
            latent_goal = self.visual_goal(perceptual_emb[:, -1])
            if self.state_recons:
                state_recon_loss = self.perceptual_encoder.state_reconstruction_loss()
                self.log(f"val/proprio_loss_{self.modality_scope}", state_recon_loss, sync_dist=True)

            robot_obs = dataset_batch["state_info"]["robot_obs"]
            actions = dataset_batch["actions"]
            empty_plan = torch.empty((dataset_batch["actions"].shape[0]), 0).to(self.device)
            action_loss, sample_act = self.action_decoder.loss_and_act(  # type:  ignore
                empty_plan, perceptual_emb, latent_goal, actions, robot_obs
            )
            mae = torch.nn.functional.l1_loss(
                sample_act[..., :-1], actions[..., :-1], reduction="none"
            )  # (batch, seq, 6)
            mae = torch.mean(mae, 1)  # (batch, 6)
            # gripper action
            gripper_discrete = sample_act[..., -1]
            gt_gripper_act = actions[..., -1]
            m = gripper_discrete > 0
            gripper_discrete[m] = 1
            gripper_discrete[~m] = -1
            gripper_sr = torch.mean((gt_gripper_act == gripper_discrete).float())

            val_total_act_loss += action_loss
            mae_mean = mae.mean()
            pos_mae = mae[..., :3].mean()
            orn_mae = mae[..., 3:6].mean()
            self.log(f"val_total_mae/{self.modality_scope}_total_mae", mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{self.modality_scope}_pos_mae", pos_mae, sync_dist=True)
            self.log(f"val_orn_mae/{self.modality_scope}_orn_mae", orn_mae, sync_dist=True)
            self.log(f"val_act/{self.modality_scope}_act_loss", action_loss, sync_dist=True)
            self.log(f"val_grip/{self.modality_scope}_grip_sr", gripper_sr, sync_dist=True)
            self.log(
                "val_act/action_loss",
                val_total_act_loss / len(self.trainer.datamodule.modalities),  # type:ignore
                sync_dist=True,
            )
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]

        return output

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.latent_goal = None

    def step(self, obs, goal):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        with torch.no_grad():
            if self.latent_goal is None:
                if isinstance(goal, str):
                    embedded_lang = torch.from_numpy(self.lang_embeddings[goal]).to(self.device).squeeze(0).float()
                    self.latent_goal = self.language_goal(embedded_lang)
                else:
                    imgs = {
                        k: torch.cat([v, goal["rgb_obs"][k]], dim=1) for k, v in obs["rgb_obs"].items()
                    }  # (1, 2, C, H, W)
                    depth_imgs = {k: torch.cat([v, goal["depth_obs"][k]], dim=1) for k, v in obs["depth_obs"].items()}
                    state = torch.cat([obs["robot_obs"], goal["robot_obs"]], dim=1)
                    perceptual_emb = self.perceptual_encoder(imgs, depth_imgs, state)
                    self.latent_goal = self.visual_goal(perceptual_emb[:, -1])

            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["robot_obs"])
            empty_plan = torch.empty(1, 0).to(self.device)
            action = self.action_decoder.act(
                empty_plan, perceptual_emb, self.latent_goal, obs["robot_obs_raw"]
            )  # type:  ignore
            return action
