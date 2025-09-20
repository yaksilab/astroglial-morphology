from cellpose.models import CellposeModel
from suite2p.io import BinaryFile
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class Segmentation:
    def __init__(
        self,
        model_path: str = r"..\models\CP3_S4_0-1_0-0001_10000",
    ) -> None:
        self.model = CellposeModel(
            gpu=False,
            pretrained_model=model_path,  # pyright: ignore[reportArgumentType]
        )

    def segment_img(self, img, resample: bool = False, diameter: int = 200):
        masks, _, _ = self.model.eval(
            x=img,
            flow_threshold=1.0,
            cellprob_threshold=0.0,
            diameter=diameter,
            augment=True,
            resample=resample,
        )
        return masks

    def segment_binaryfile(
        self,
        ops_path: str,
        bin_file_path: str,
        resample: bool = False,
        frame_jump: int = 1,
    ):
        ops = np.load(ops_path, allow_pickle=True).item()
        Lx, Ly = ops["Lx"], ops["Ly"]
        nframes = ops["nframes"]
        data = BinaryFile(Ly=Ly, Lx=Lx, filename=bin_file_path, n_frames=nframes).data
        logger.info(
            f"Starting segmentation of {nframes} frames with frame jump {frame_jump}"
        )
        for i in range(0, nframes, frame_jump):
            img = data[i]
            logger.info(f"Segmenting frame {i+1}/{nframes}")
            masks = self.segment_img(img, resample=resample)
            yield i, masks
        logger.info("Segmentation completed")
