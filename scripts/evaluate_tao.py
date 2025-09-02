import argparse
import logging
from pathlib import Path
import sys

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from pytorch_lightning import seed_everything

from calvin_agent.evaluation.evaluate_policy_visual_goal import evaluate_policy
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint

from taowm.evaluation.utils import get_model_env_taskchecker_datamodule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    return checkpoint.stem.split("=")[1]


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with visual goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading the model and visual goals
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument("--start_end_tasks", type=str, help="Path to the start_end_tasks.yaml file.")
    parser.add_argument(
        "--num_sequences", type=int, default=100, help="Number of sequences to evaluate. Default is 100."
    )
    parser.add_argument(
        "--num_tasks_per_sequence", type=int, default=5, help="Number of tasks per sequence. Default is 5."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    assert "train_folder" in args, "train_folder is needed for finding the checkpoints and configs"
    assert "start_end_tasks" in args, "start_end_tasks.yaml is needed for obtain visual goals"
    checkpoints = []
    if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
        logger.info("Evaluating model with last checkpoint.")
        breakpoint()
        checkpoints = [get_last_checkpoint(Path(args.train_folder))]
    elif args.checkpoints is not None:
        logger.info(f"Evaluating model with checkpoints {args.checkpoints}.")
        checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
    elif args.checkpoints is None and args.last_k_checkpoints is not None:
        logger.info(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
        checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
    elif args.checkpoint is not None:
        checkpoints = [Path(args.checkpoint)]

    # prepare model, env, task_checker, and datamodule, then evaluate
    env = None
    for checkpoint in checkpoints:
        logger.info(f"Evaluating checkpoint {checkpoint}")
        model, env, task_checker, datamodule = get_model_env_taskchecker_datamodule(
            args.train_folder,
            args.dataset_path,
            checkpoint,
            env=env,
            device_id=args.device,
        )
        evaluate_policy(
            model=model,
            env=env,
            task_checker=task_checker,
            start_end_tasks_file=args.start_end_tasks,
            val_dataset_dir=datamodule.val_datasets["vis"].abs_datasets_dir,
            num_sequences=args.num_sequences,
            num_tasks_per_sequence=args.num_tasks_per_sequence,
            eval_log_dir=args.eval_log_dir,
        )


if __name__ == "__main__":
    main()
