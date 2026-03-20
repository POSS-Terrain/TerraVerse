from __future__ import annotations

from typing import Callable, Dict, Sequence

from utils.mask_to_patch_acdc import run as run_acdc
from utils.mask_to_patch_coco_stuff import run as run_coco_stuff
from utils.mask_to_patch_deepscene import run as run_deepscene
from utils.mask_to_patch_fcdd import run as run_fcdd
from utils.mask_to_patch_goose import run as run_goose
from utils.mask_to_patch_goose_ex import run as run_goose_ex
from utils.mask_to_patch_idd import run as run_idd
from utils.mask_to_patch_kitti_360 import run as run_kitti_360
from utils.mask_to_patch_orad_3d import run as run_orad_3d
from utils.mask_to_patch_orad_3d_label import run as run_orad_3d_label
from utils.mask_to_patch_orfd import run as run_orfd
from utils.mask_to_patch_rellis import run as run_rellis
from utils.mask_to_patch_rtk import run as run_rtk
from utils.mask_to_patch_rugd import run as run_rugd
from utils.mask_to_patch_tas500 import run as run_tas500
from utils.mask_to_patch_vast import run as run_vast
from utils.mask_to_patch_wildscenes import run as run_wildscenes
from utils.mask_to_patch_ycor import run as run_ycor


EXCLUDED_DATASETS = {"Jackal", "TerraPOSS", "RSCD"}
HANDLERS: Dict[str, Callable[[Sequence[str] | None], None]] = {
    "ACDC": run_acdc,
    "COCO-Stuff": run_coco_stuff,
    "DeepScene": run_deepscene,
    "FCDD": run_fcdd,
    "GOOSE": run_goose,
    "GOOSE-Ex": run_goose_ex,
    "IDD": run_idd,
    "KITTI-360": run_kitti_360,
    "ORAD-3D": run_orad_3d,
    "ORAD-3D-Label": run_orad_3d_label,
    "ORFD": run_orfd,
    "RELLIS": run_rellis,
    "RTK": run_rtk,
    "RUGD": run_rugd,
    "TAS500": run_tas500,
    "VAST": run_vast,
    "WildScenes": run_wildscenes,
    "YCOR": run_ycor,
}
