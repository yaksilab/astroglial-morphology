from typing import Generator
from cellpose.models import CellposeModel
from suite2p.io import BinaryFile
from cellpose.io import masks_flows_to_seg
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class Segmentation:
    """
    Segmentation class using the Cellpose model.
    """

    def __init__(
        self,
        model_path: str = r"C:\Users\javid.rezai\YaksiLab\duygu\astroglial-morphology\astroglial-morphology\src\models\CP3_S4_0-1_0-0001_10000",
        gpu: bool = False,
    ) -> None:
        self.model = CellposeModel(
            gpu=gpu,
            pretrained_model=model_path,  # pyright: ignore[reportArgumentType]
        )
        self.default_eval_params = {
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "diameter": None,
            "augment": True,
            "resample": True,
            "min_size": 80,
        }
        self.default_eval_params["normalize"] = {
            "lowhigh": None,
            "percentile": [1.0, 99.0],
            "normalize": True,
            "norm3D": True,
            "sharpen_radius": 0,
            "smooth_radius": 0,
            "tile_norm_blocksize": 0,
            "tile_norm_smooth3D": 1,
            "invert": False,
        }

    def segment_img(
        self,
        img: np.ndarray,
        save_file_name: str = "image_masks",
        **kwargs,
    ) -> np.ndarray:
        """
        Segment an image using the Cellpose model.

        Args:
            img: The image to segment.
            **kwargs: Additional parameters to pass to the Cellpose model.
            Default parameters are:
            - flow_threshold: 0.4
            - cellprob_threshold: 0.0
            - diameter: None
            - augment: True
            - resample: False
            - normalize: {
                - lowhigh: None
                - percentile: [1.0, 99.0]
                - normalize: True
                - norm3D: True
                - sharpen_radius: 0
                - smooth_radius: 0
                - tile_norm_blocksize: 0
                - tile_norm_smooth3D: 1
                - invert: False
            }

            All parameters are passed to the Cellpose model.

        Returns:
            The masks of the segmented image.
        """

        default_eval_params = self.default_eval_params.copy()
        default_eval_params.update(kwargs)

        model_eval_params = default_eval_params.copy()
        model_eval_params.update(kwargs)

        logger.info(f"Segmenting with Segmentation parameters: {model_eval_params}")

        masks, flows, _ = self.model.eval(
            x=img,
            **model_eval_params,  # pyright: ignore[reportArgumentType]
        )
        

        masks_flows_to_seg(img, masks, flows, save_file_name)
        return masks

    def segment_binaryfile(
        self,
        ops_path: str,
        bin_file_path: str,
        frame_jump: int = 1,
        **kwargs,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Segment a binary file using the Cellpose model.

        Args:
            ops_path: The path to the ops file.
            bin_file_path: The path to the binary file.
            frame_jump: The frame jump to use for segmentation.
            **kwargs: Additional parameters to pass to the Cellpose model.

        Returns:
            A generator of tuples containing the frame index and the masks of the segmented image.
        """
        ops = np.load(ops_path, allow_pickle=True).item()
        Lx, Ly = ops["Lx"], ops["Ly"]
        nframes = ops["nframes"]
        data = BinaryFile(Ly=Ly, Lx=Lx, filename=bin_file_path, n_frames=nframes).data

        model_eval_params = self.default_eval_params.copy()
        model_eval_params.update(kwargs)
        logger.info(
            f"Starting segmentation of {nframes} frames with frame jump {frame_jump}"
        )
        for i in range(0, nframes, frame_jump):
            img = data[i]
            logger.info(f"Segmenting frame {i+1}/{nframes}")
            masks = self.segment_img(img, **model_eval_params)
            yield i, masks
        logger.info("Segmentation completed")
