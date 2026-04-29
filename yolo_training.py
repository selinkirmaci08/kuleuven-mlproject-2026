import subprocess
import os
from pathlib import Path


class YOLOTrainer:
    """
    A wrapper class to run YOLO detection training via the CLI.

    Example usage:
        trainer = YOLOTrainer(
            data="dataset/clean/dataset.yaml",
            model="yolo11n.pt",
            epochs=50,
            imgsz=640
        )
        trainer.train()
    """

    def __init__(
        self,
        data: str,
        model: str = "yolo11n.pt",
        epochs: int = 50,
        imgsz: int = 640,
        task: str = "detect",
        extra_args: dict = None,
    ):
        """
        Initialize the YOLOTrainer.

        Args:
            data:       Path to the dataset YAML file.
            model:      YOLO model weights file (e.g. 'yolo11n.pt').
            epochs:     Number of training epochs.
            imgsz:      Input image size (pixels).
            task:       YOLO task type ('detect', 'segment', 'classify').
            extra_args: Optional dict of additional YOLO CLI arguments,
                        e.g. {'batch': 16, 'device': 0}.
        """
        self.data = data
        self.model = model
        self.epochs = epochs
        self.imgsz = imgsz
        self.task = task
        self.extra_args = extra_args or {}

    def _build_command(self) -> list[str]:
        """Build the YOLO CLI command as a list of arguments."""
        cmd = [
            "yolo",
            self.task,
            "train",
            f"data={self.data}",
            f"model={self.model}",
            f"epochs={self.epochs}",
            f"imgsz={self.imgsz}",
        ]
        for key, value in self.extra_args.items():
            cmd.append(f"{key}={value}")
        return cmd

    def train(self, cwd: str = None) -> int:
        """
        Run the YOLO training command.

        Args:
            cwd: Working directory to run the command from.
                 Defaults to the current directory.

        Returns:
            The process return code (0 = success).
        """
        cmd = self._build_command()
        print("Running command:")
        print(" ".join(cmd))
        print("-" * 50)

        result = subprocess.run(
            cmd,
            cwd=cwd or os.getcwd(),
            check=False,  # Don't raise on non-zero exit; handle it below
        )

        if result.returncode != 0:
            print(f"\nTraining failed with return code {result.returncode}.")
        else:
            print("\nTraining completed successfully.")

        return result.returncode


# ── Quick-start ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = YOLOTrainer(
        data=r"dataset\clean\dataset.yaml",
        model="yolo11n.pt",
        epochs=50,
        imgsz=640,
        # Optional extras, e.g.:
        # extra_args={"batch": 16, "device": 0, "workers": 4},
    )
    trainer.train()