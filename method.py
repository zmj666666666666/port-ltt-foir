from __future__ import annotations

from pathlib import Path
import time
import json
from dataclasses import dataclass, field
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

from ultralytics import YOLO

# ===================== 可自定义参数 =====================
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
VIDEO_PATH = ""
TRACKER_CFG_PATH = PROJECT_ROOT / "cfg" / "my_botsort.yaml"
SCENE_DIVISION_PATH = PROJECT_ROOT / "scene-division" / "scene-division.json"

START_FRAME = 0
DET_INTERVAL = 2
CONF_THRES_TRACK = 0.2
CONF_THRES_DISPLAY = 0.25
NMS_IOU = 0.5
IMGSZ = 960
INTERVAL_SECONDS = 0.0
SHOW_LABEL = True
SHOW_CONF = True
LINE_THICKNESS = 2
DISPLAY_MODE = "opencv"
SHOW_WHITEBOARD = True
ICON_RADIUS = 30
ICON_TEXT_SCALE = 0.45
DISPLAY_SCALE = 0.75
AUTO_FIT_SCREEN = True
SCREEN_MARGIN = 120
USE_HALF = torch.cuda.is_available()
DEVICE = 0 if torch.cuda.is_available() else "cpu"
TRACK_MAX_MISSING = 90
TRACK_MIN_HITS = 2
TRACK_PREDICT_RENDER = 10
STOP_SPEED_DIAG_RATIO = 0.005
STOP_MOVE_DIAG_RATIO = 0.006
STOP_CONFIRM_FRAMES = 3
AFTER_TRACK_IOU = 0.6
AFTER_TRACK_CENTER_DIST_RATIO = 0.35
SEMANTIC_HISTORY_LEN = 6
SEMANTIC_CONFIRM_FRAMES = 2
SEMANTIC_TURN_MIN_SPEED = 1.2
SEMANTIC_TURN_ANGLE_THRESH_DEG = 18.0
SEMANTIC_WAIT_CONFIRM_FRAMES = 8
SEMANTIC_UNLOADING_CONFIRM_FRAMES = 36
SEMANTIC_AREA_STABILITY_THRESH = 0.12
SEMANTIC_NORMAL_MIN_SPEED = 1.1
STATIC_LOCK_IOU = 0.8
STATIC_LOCK_CENTER_RATIO = 0.15
STATIC_RELEASE_COOLDOWN = 10
STATIC_OUTLIER_DIAG_RATIO = 0.5 #当检测框中心点位移大于这个值时，认为是其他目标干扰造成的剧烈抖动，忽略
NEW_TRACK_ENTRANCE_GATE_FRAMES = 30
REID_IOU_THRESH = 0.8
REID_IOU_MIN = 0.3 #自适应门槛的下限
REID_MISSING_DECAY_FRAMES = 120 #门槛从0.8线性衰减到0.3的帧数
GHOST_MAX_AGE = 60 #幽灵轨迹额外存活帧数

# -- OSNet ReID 参数 --
REID_MODEL_PATH = PROJECT_ROOT / "models" / "osnet_x1_0_imagenet.pth"
REID_FEATURE_HISTORY_LEN = 5
REID_IMAGE_SIZE = (256, 128)
REID_APPEARANCE_WEIGHT_INIT = 0.75
REID_APPEARANCE_WEIGHT_MIN = 0.15
REID_APPEARANCE_DECAY_FRAMES = 120
REID_COSINE_THRESH = 0.4
REID_APPEARANCE_EMA_ALPHA = 0.6

SEMANTIC_DISPLAY_NAMES = {
    "UNKNOWN": "未知",
    "NORMAL_DRIVING": "正常行驶",
    "TURNING": "转弯",
    "STOP_WAITING": "停车等待",
    "UNLOADING": "装卸货",
}
SEMANTIC_ABBR = {
    "UNKNOWN": "UN",
    "NORMAL_DRIVING": "NM",
    "TURNING": "TR",
    "STOP_WAITING": "SW",
    "UNLOADING": "UL",
}
SEMANTIC_TRACK_POLICY = {
    "UNKNOWN": {
        "predict_scale": 0.85, #非检测帧时，轨迹按当前速度外推的倍率
        "max_missing_scale": 1.0, #允许连续丢失多少帧后删除轨迹的倍率
        "max_predict_render_scale": 1.0, #预测框允许继续显示多少帧的倍率
        "display_conf_scale": 1.0,#渲染时最低置信度阈值的倍率
        "suppress_iou": AFTER_TRACK_IOU, #重叠抑制时的 IoU 门槛。会选多个目标框中最高那个
        "suppress_center_ratio": AFTER_TRACK_CENTER_DIST_RATIO, #重叠抑制时的中心距离门槛比例。先过了iou判断才会进中心距离判断
    },
    "NORMAL_DRIVING": {
        "predict_scale": 1.0,
        "max_missing_scale": 1.0,
        "max_predict_render_scale": 1.0,
        "display_conf_scale": 1.0,
        "suppress_iou": AFTER_TRACK_IOU,
        "suppress_center_ratio": AFTER_TRACK_CENTER_DIST_RATIO,
    },
    "TURNING": {
        "predict_scale": 0.6,
        "max_missing_scale": 1.0,
        "max_predict_render_scale": 1.5,
        "display_conf_scale": 0.95,
        "suppress_iou": 0.72,
        "suppress_center_ratio": 0.45,
    },
    "STOP_WAITING": {
        "predict_scale": 0.0,
        "max_missing_scale": 1.0,
        "max_predict_render_scale": 9.0,
        "display_conf_scale": 0.85,
        "suppress_iou": 0.8,
        "suppress_center_ratio": 0.35,
        "static_lock_release_frames": 5,
    },
    "UNLOADING": {
        "predict_scale": 0.0,
        "max_missing_scale": 1.0,
        "max_predict_render_scale": 9.0,
        "display_conf_scale": 0.85,
        "suppress_iou": 0.8,
        "suppress_center_ratio": 0.35,
        "static_lock_release_frames": 15,
    },
}

# ========================================================


def _load_scene_division() -> dict:
    """加载场景划分配置文件。

    Returns:
        场景划分数据字典，加载失败时返回空字典。
    """
    try:
        with open(SCENE_DIVISION_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _match_scene_name(video_path: Path, scene_keys: list[str]) -> str | None:
    """根据视频文件名匹配场景名称。

    Args:
        video_path: 视频文件路径。
        scene_keys: 场景划分中的所有场景键名。

    Returns:
        匹配到的场景名称，未匹配时返回 None。
    """
    stem = video_path.stem
    for key in scene_keys:
        if stem.startswith(key):
            return key
    return None


def _load_precision_zone_polygon(video_path: Path) -> np.ndarray | None:
    """加载视频对应的精度区域多边形。

    Args:
        video_path: 视频文件路径。

    Returns:
        精度区域多边形顶点数组 (N, 2)，未找到时返回 None。
    """
    scene_data = _load_scene_division()
    if not scene_data:
        return None
    scene_name = _match_scene_name(video_path, list(scene_data.keys()))
    if scene_name is None:
        return None
    for region in scene_data[scene_name]:
        if region.get("label") == "precision_zone":
            return np.array(region["points"], dtype=np.float32)
    return None


def _load_non_road_polygons(video_path: Path) -> list[np.ndarray]:
    """加载视频对应的所有非道路区域多边形。

    Args:
        video_path: 视频文件路径。

    Returns:
        非道路区域多边形列表，每个元素为 (N, 2) 的顶点数组。未找到时返回空列表。
    """
    scene_data = _load_scene_division()
    if not scene_data:
        return []
    scene_name = _match_scene_name(video_path, list(scene_data.keys()))
    if scene_name is None:
        return []
    polygons: list[np.ndarray] = []
    for region in scene_data[scene_name]:
        if region.get("label") == "non-road":
            polygons.append(np.array(region["points"], dtype=np.float32))
    return polygons


def _load_handling_polygons(video_path: Path) -> list[np.ndarray]:
    """加载视频对应的所有装卸货区域多边形。

    Args:
        video_path: 视频文件路径。

    Returns:
        装卸货区域多边形列表，每个元素为 (N, 2) 的顶点数组。未找到时返回空列表。
    """
    scene_data = _load_scene_division()
    if not scene_data:
        return []
    scene_name = _match_scene_name(video_path, list(scene_data.keys()))
    if scene_name is None:
        return []
    polygons: list[np.ndarray] = []
    for region in scene_data[scene_name]:
        if region.get("label") == "handling":
            polygons.append(np.array(region["points"], dtype=np.float32))
    return polygons


def _load_entrance_polygons(video_path: Path) -> list[np.ndarray]:
    """加载视频对应的所有入口区多边形。

    Args:
        video_path: 视频文件路径。

    Returns:
        入口区多边形列表，每个元素为 (N, 2) 的顶点数组。未找到时返回空列表。
    """
    scene_data = _load_scene_division()
    if not scene_data:
        return []
    scene_name = _match_scene_name(video_path, list(scene_data.keys()))
    if scene_name is None:
        return []
    polygons: list[np.ndarray] = []
    for region in scene_data[scene_name]:
        if region.get("label") == "entrance":
            polygons.append(np.array(region["points"], dtype=np.float32))
    return polygons


def _is_bbox_in_non_road(bbox: np.ndarray, non_road_polygons: list[np.ndarray]) -> bool:
    """判断目标框是否完全位于非道路区域内。

    目标框的顶部中心点和底部中心点均在非道路区域（任一多边形）内时，
    判定为完全处于非道路区。

    Args:
        bbox: 目标框 [x1, y1, x2, y2]。
        non_road_polygons: 非道路区域多边形列表。

    Returns:
        目标框完全在非道路区域内返回 True。
    """
    if not non_road_polygons:
        return False
    x1, y1, x2, y2 = bbox
    top_center = ((x1 + x2) / 2.0, float(y1))
    bottom_center = ((x1 + x2) / 2.0, float(y2))
    for polygon in non_road_polygons:
        if _point_in_polygon(top_center, polygon) and _point_in_polygon(bottom_center, polygon):
            return True
    return False


def _is_bbox_in_handling_zone(bbox: np.ndarray, handling_polygons: list[np.ndarray]) -> bool:
    """判断目标框是否位于装卸货区域内。

    目标框的顶部中心点或底部中心点位于装卸货区域（任一多边形）内时，
    即判定为在装卸货区域。

    Args:
        bbox: 目标框 [x1, y1, x2, y2]。
        handling_polygons: 装卸货区域多边形列表。

    Returns:
        目标框在装卸货区域内返回 True。
    """
    if not handling_polygons:
        return False
    x1, y1, x2, y2 = bbox
    top_center = ((x1 + x2) / 2.0, float(y1))
    bottom_center = ((x1 + x2) / 2.0, float(y2))
    for polygon in handling_polygons:
        if _point_in_polygon(top_center, polygon) or _point_in_polygon(bottom_center, polygon):
            return True
    return False


def _is_bbox_in_entrance_zone(bbox: np.ndarray, entrance_polygons: list[np.ndarray]) -> bool:
    """判断目标框是否位于入口区域内。

    目标框的顶部中心点或底部中心点位于入口区域（任一多边形）内时，
    即判定为在入口区域。

    Args:
        bbox: 目标框 [x1, y1, x2, y2]。
        entrance_polygons: 入口区域多边形列表。

    Returns:
        目标框在入口区域内返回 True。
    """
    if not entrance_polygons:
        return False
    x1, y1, x2, y2 = bbox
    top_center = ((x1 + x2) / 2.0, float(y1))
    bottom_center = ((x1 + x2) / 2.0, float(y2))
    for polygon in entrance_polygons:
        if _point_in_polygon(top_center, polygon) or _point_in_polygon(bottom_center, polygon):
            return True
    return False


def _point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """射线法判断点是否在多边形内。

    Args:
        point: 待检测点 (x, y)。
        polygon: 多边形顶点数组 (N, 2)。

    Returns:
        点在多边形内返回 True。
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _is_bbox_in_precision_zone(bbox: np.ndarray, precision_polygon: np.ndarray | None) -> bool:
    """判断目标框是否在精度区域内。

    检查目标框顶部中心点和底部中心点，任意一点在精度区域内即认为目标在区域内。

    Args:
        bbox: 目标框 [x1, y1, x2, y2]。
        precision_polygon: 精度区域多边形，None 时全域视为精度区域（兼容无配置场景）。

    Returns:
        目标在精度区域内返回 True。
    """
    if precision_polygon is None:
        return True
    x1, y1, x2, y2 = bbox
    top_center = ((x1 + x2) / 2.0, float(y1))
    bottom_center = ((x1 + x2) / 2.0, float(y2))
    return _point_in_polygon(top_center, precision_polygon) or _point_in_polygon(
        bottom_center, precision_polygon
    )


def _gated_semantic_state(track: TrackState, precision_polygon: np.ndarray | None) -> str:
    """获取精度区域门控后的语义状态。

    非精度区域内的目标回退为 UNKNOWN，使追踪策略使用默认参数，
    但 track.semantic_state 保持不变，语义标签照常显示。

    Args:
        track: 轨迹状态。
        precision_polygon: 精度区域多边形，None 时不过滤。

    Returns:
        用于追踪策略决策的语义状态。
    """
    if precision_polygon is None:
        return track.semantic_state
    if _is_bbox_in_precision_zone(track.bbox, precision_polygon):
        return track.semantic_state
    return "UNKNOWN"


def _gated_is_static_semantic(track: TrackState, precision_polygon: np.ndarray | None) -> bool:
    """精度区域门控的静止语义判断。

    非精度区域内的目标不应用静态锁定逻辑。
    释放冷却期内禁止重新冻结。

    Args:
        track: 轨迹状态。
        precision_polygon: 精度区域多边形，None 时不过滤。

    Returns:
        是否应作为静止语义处理。
    """
    if track.static_release_cooldown > 0:
        return False
    if precision_polygon is None:
        return _is_static_semantic_state(track.semantic_state)
    if not _is_bbox_in_precision_zone(track.bbox, precision_polygon):
        return False
    return _is_static_semantic_state(track.semantic_state)


# ============================================================================
# OSNet 特征提取器
# ============================================================================


class OSNetFeatureExtractor:
    """OSNet 外观特征提取器。

    从 YOLO 检测框裁剪出的目标图像中提取 L2 归一化的 512 维外观嵌入向量，
    用于多目标跟踪中的外观辅助匹配和遮挡后的重识别。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | int = "cpu",
        use_half: bool = False,
        image_size: tuple[int, int] = (256, 128),
    ):
        """初始化 OSNet 特征提取器。

        通过 torchreid 构建 OSNet 架构，从 model_path 加载预训练权重。
        """
        self.image_size = image_size
        self.device = torch.device(device)
        self.use_half = use_half and self.device.type == "cuda"

        self.model = self._build_osnet(model_path)

        self.model.to(self.device)
        if self.use_half:
            self.model.half()
        self.model.eval()

        # 图像预处理: BGR→RGB, resize, normalize
        h, w = image_size
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def _build_osnet(model_path: str | Path | None) -> nn.Module:
        """构建 OSNet 模型，从指定路径加载权重。"""
        import torchreid

        backbone = torchreid.models.osnet_x1_0(pretrained=False)

        if model_path is not None and Path(model_path).exists():
            print(f"OSNet 从指定路径加载权重: {model_path}")
            state_dict = torch.load(
                str(model_path), map_location="cpu", weights_only=True
            )
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state[k] = v
            model_dict = backbone.state_dict()
            filtered_dict = {}
            for k, v in new_state.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
            model_dict.update(filtered_dict)
            backbone.load_state_dict(model_dict, strict=False)
            matched = len(filtered_dict)
            total = len(model_dict)
            print(f"OSNet 权重加载完成: {matched}/{total} 个参数匹配")
        else:
            print("未找到指定权重文件，OSNet 使用随机初始化（ReID 效果受限）")

        class _NormalizedWrapper(nn.Module):
            """L2 归一化包装器，确保 OSNet 输出经 L2 归一化。"""
            def __init__(self, backbone: nn.Module):
                super().__init__()
                self.backbone = backbone

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.normalize(self.backbone(x), p=2, dim=1)

        return _NormalizedWrapper(backbone)

    @torch.no_grad()
    def extract_feature(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray | None:
        """从单帧的单个检测框中提取外观特征。"""
        crop = self._crop_bbox(frame, bbox)
        if crop is None:
            return None
        return self._extract_from_crop(crop)

    @torch.no_grad()
    def extract_features_batch(
        self, frame: np.ndarray, bboxes: list[np.ndarray]
    ) -> list[np.ndarray | None]:
        """从单帧的多个检测框中批量提取外观特征。

        使用批量推理以提高 GPU 利用率。
        """
        crops = []
        valid_indices = []
        for i, bbox in enumerate(bboxes):
            crop = self._crop_bbox(frame, bbox)
            if crop is not None:
                crops.append(crop)
                valid_indices.append(i)

        if len(crops) == 0:
            return [None] * len(bboxes)

        # 批量预处理
        tensors = []
        for crop in crops:
            t = self.transform(crop)
            tensors.append(t)
        batch = torch.stack(tensors, dim=0).to(self.device)
        if self.use_half:
            batch = batch.half()

        # 批量推理
        features_t = self.model(batch)
        features = features_t.cpu().numpy()

        # 按原始顺序填充结果
        result: list[np.ndarray | None] = [None] * len(bboxes)
        for idx, feat in zip(valid_indices, features):
            result[idx] = feat

        return result

    def _crop_bbox(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray | None:
        """从帧中裁剪目标框区域。"""
        h, w = frame.shape[:2]
        x1 = int(max(0, min(bbox[0], w - 1)))
        y1 = int(max(0, min(bbox[1], h - 1)))
        x2 = int(max(0, min(bbox[2], w - 1)))
        y2 = int(max(0, min(bbox[3], h - 1)))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _extract_from_crop(self, crop: np.ndarray) -> np.ndarray | None:
        """从单个裁剪块中提取归一化特征。"""
        if crop.size == 0:
            return None
        try:
            t = self.transform(crop).unsqueeze(0).to(self.device)
            if self.use_half:
                t = t.half()
            feat = self.model(t)
            return feat.squeeze(0).cpu().numpy()
        except Exception:
            return None


# ============================================================================
# 余弦相似度工具函数
# ============================================================================


def _cosine_similarity(
    feat_a: np.ndarray, feat_b: np.ndarray
) -> float:
    """计算两个特征向量的余弦相似度。"""
    norm_a = float(np.linalg.norm(feat_a))
    norm_b = float(np.linalg.norm(feat_b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(feat_a, feat_b) / (norm_a * norm_b))


def _cosine_similarity_max(
    query_feat: np.ndarray,
    gallery_feats: deque,
) -> float:
    """计算查询特征与特征库中所有特征的最大余弦相似度。"""
    if len(gallery_feats) == 0:
        return 0.0
    best = 0.0
    for feat in gallery_feats:
        sim = _cosine_similarity(query_feat, feat)
        if sim > best:
            best = sim
    return best


@dataclass
class TrackedBox:
    """保存单个跟踪目标的渲染信息。

    Args:
        bbox: 目标框，格式为 [x1, y1, x2, y2]。
        cls_id: 类别 ID。
        cls_name: 类别名称。
        conf: 置信度。
        track_id: 跟踪 ID。
    """

    bbox: np.ndarray
    cls_id: int
    cls_name: str
    conf: float
    track_id: int | None


@dataclass
class TrackState:
    """保存单条 BoT-SORT 轨迹的连续状态。

    Args:
        track_id: 轨迹 ID。
        cls_id: 类别 ID。
        cls_name: 类别名称。
        bbox: 当前框，格式为 [x1, y1, x2, y2]。
        conf: 当前置信度。
        hits: 成功命中次数。
        age: 轨迹存活帧数。
        missing_frames: 连续丢失帧数。
        vx: x 方向速度。
        vy: y 方向速度。
        stop_count: 连续静止判定计数。
        motion_state: 运动状态，取值为 "MOVE" 或 "STOPPED"。
        is_predicted: 当前是否为预测框。
        center_history: 轨迹中心点历史。
        speed_history: 轨迹速度历史。
        heading_history: 轨迹航向角历史。
        area_history: 轨迹框面积历史。
        semantic_state: 当前语义状态。
        semantic_conf: 当前语义置信度。
        semantic_candidate: 最近一次语义候选状态。
        semantic_candidate_count: 连续候选命中次数。
        anchor_bbox: 静止目标锚点框。
        static_lock: 当前是否处于静止冻结状态。
        static_mismatch_count: 连续观测偏离锚点的次数。
        static_release_cooldown: 释放后冷却帧数，冷却期内禁止重新冻结。
    """

    track_id: int
    cls_id: int
    cls_name: str
    bbox: np.ndarray
    conf: float
    display_id: int | None = None
    hits: int = 1
    age: int = 1
    missing_frames: int = 0
    vx: float = 0.0
    vy: float = 0.0
    stop_count: int = 0
    motion_state: str = "MOVE"
    is_predicted: bool = False
    center_history: deque = field(default_factory=lambda: deque(maxlen=SEMANTIC_HISTORY_LEN))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=SEMANTIC_HISTORY_LEN))
    heading_history: deque = field(default_factory=lambda: deque(maxlen=SEMANTIC_HISTORY_LEN))
    area_history: deque = field(default_factory=lambda: deque(maxlen=SEMANTIC_HISTORY_LEN))
    semantic_state: str = "UNKNOWN"
    semantic_conf: float = 0.0
    semantic_candidate: str = "UNKNOWN"
    semantic_candidate_count: int = 0
    anchor_bbox: np.ndarray | None = None
    static_lock: bool = False
    static_mismatch_count: int = 0
    static_release_cooldown: int = 0
    last_observed_bbox: np.ndarray | None = None
    ghost_age: int = 0
    appearance_features: deque = field(
        default_factory=lambda: deque(maxlen=REID_FEATURE_HISTORY_LEN)
    )
    last_appearance_feature: np.ndarray | None = None


def _class_color(class_id: int) -> tuple[int, int, int]:
    """按类别返回固定颜色。

    Args:
        class_id: 类别 ID。

    Returns:
        BGR 颜色元组。
    """
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 215, 0),
        (186, 85, 211),
        (72, 209, 204),
        (220, 20, 60),
        (0, 191, 255),
    ]
    return palette[class_id % len(palette)]


def _draw_whiteboard(
    frame_shape: tuple[int, int, int],
    boxes_xyxy: list[list[float]],
    classes: list[int],
    names: dict[int, str],
    semantic_states: list[str] | None = None,
    precision_polygon: np.ndarray | None = None,
):
    """绘制白板视图。

    Args:
        frame_shape: 帧尺寸。
        boxes_xyxy: 目标框列表。
        classes: 类别 ID 列表。
        names: 类别名称映射。
        semantic_states: 每个目标对应的语义状态列表。
        precision_polygon: 精度区域多边形顶点数组 (N, 2)，为 None 时不绘制。

    Returns:
        白板图像。
    """
    h, w = frame_shape[:2]
    board = np.full((h, w, 3), 255, dtype=np.uint8)

    if precision_polygon is not None:
        overlay = board.copy()
        pts = precision_polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], (215, 235, 215))
        board = cv2.addWeighted(board, 0.92, overlay, 0.08, 0)
        cv2.polylines(board, [pts], isClosed=True, color=(100, 175, 100), thickness=1, lineType=cv2.LINE_AA)
        centroid = precision_polygon.mean(axis=0).astype(int)
        (tw, th), _ = cv2.getTextSize("precision_zone", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_org = (max(4, int(centroid[0]) - tw // 2), max(14, int(centroid[1]) + th // 2))
        cv2.putText(
            board,
            "precision_zone",
            label_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (90, 140, 90),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(board, "Whiteboard", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 90, 90), 2, cv2.LINE_AA)
    if semantic_states is None:
        semantic_states = ["UNKNOWN"] * len(boxes_xyxy)

    for box, cls_id, semantic_state in zip(boxes_xyxy, classes, semantic_states):
        x1, y1, x2, y2 = box
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        cx = int(round(x1 + bw / 2.0))
        cy = int(round(y1 + bh / 3.0))
        cx = min(max(cx, ICON_RADIUS), w - ICON_RADIUS)
        cy = min(max(cy, ICON_RADIUS), h - ICON_RADIUS)

        color = _class_color(cls_id)
        cv2.circle(board, (cx, cy), ICON_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(board, (cx, cy), ICON_RADIUS, (40, 40, 40), 1, cv2.LINE_AA)

        cls_name = names.get(cls_id, str(cls_id))
        label = cls_name[:2].upper()
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, ICON_TEXT_SCALE, 1)[0]
        text_org = (cx - text_size[0] // 2, cy + text_size[1] // 2)
        cv2.putText(board, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, ICON_TEXT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

        semantic_abbr = _semantic_abbr(semantic_state)
        sem_org = (min(w - 50, cx + ICON_RADIUS + 6), max(16, cy + 5))
        cv2.putText(
            board,
            semantic_abbr,
            sem_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )

    return board


def _get_screen_size() -> tuple[int | None, int | None]:
    """获取屏幕分辨率。

    Returns:
        屏幕宽高，失败时返回 (None, None)。
    """
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        return screen_w, screen_h
    except Exception:
        return None, None


def _fit_with_letterbox(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """等比缩放并居中填充到指定尺寸。

    Args:
        image: 输入图像。
        max_w: 目标最大宽。
        max_h: 目标最大高。

    Returns:
        处理后的图像。
    """
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image

    scale = min(max_w / w, max_h / h, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
    x0 = (max_w - new_w) // 2
    y0 = (max_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _bbox_center(bbox: np.ndarray) -> tuple[float, float]:
    """计算目标框中心点。

    Args:
        bbox: 目标框，格式为 [x1, y1, x2, y2]。

    Returns:
        目标框中心点坐标。
    """
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _clip_bbox(bbox: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """将目标框裁剪到图像范围内。

    Args:
        bbox: 原始目标框。
        frame_w: 图像宽度。
        frame_h: 图像高度。

    Returns:
        裁剪后的目标框。
    """
    x1, y1, x2, y2 = bbox
    x1 = min(max(0.0, x1), frame_w - 1.0)
    y1 = min(max(0.0, y1), frame_h - 1.0)
    x2 = min(max(0.0, x2), frame_w - 1.0)
    y2 = min(max(0.0, y2), frame_h - 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _bbox_area(bbox: np.ndarray) -> float:
    """计算目标框面积。

    Args:
        bbox: 目标框，格式为 [x1, y1, x2, y2]。

    Returns:
        目标框面积。
    """
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _bbox_diagonal(bbox: np.ndarray) -> float:
    """计算目标框对角线长度。

    Args:
        bbox: 目标框，格式为 [x1, y1, x2, y2]。

    Returns:
        目标框对角线长度。
    """
    x1, y1, x2, y2 = bbox
    return float(np.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1)))


def _bbox_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """计算两个目标框的 IoU。

    Args:
        bbox_a: 第一个目标框，格式为 [x1, y1, x2, y2]。
        bbox_b: 第二个目标框，格式为 [x1, y1, x2, y2]。

    Returns:
        两个目标框的 IoU。
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(float(ax1), float(bx1))
    inter_y1 = max(float(ay1), float(by1))
    inter_x2 = min(float(ax2), float(bx2))
    inter_y2 = min(float(ay2), float(by2))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def _bbox_center_distance(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """计算两个目标框中心点之间的距离。

    Args:
        bbox_a: 第一个目标框，格式为 [x1, y1, x2, y2]。
        bbox_b: 第二个目标框，格式为 [x1, y1, x2, y2]。

    Returns:
        两个目标框中心点之间的欧式距离。
    """
    ax, ay = _bbox_center(bbox_a)
    bx, by = _bbox_center(bbox_b)
    return float(np.hypot(ax - bx, ay - by))


def _normalize_angle_deg(angle: float) -> float:
    """将角度归一化到 [-180, 180) 区间。

    Args:
        angle: 输入角度，单位为度。

    Returns:
        归一化后的角度。
    """
    return float((angle + 180.0) % 360.0 - 180.0)


def _angle_diff_deg(angle_a: float, angle_b: float) -> float:
    """计算两个角度的最小差值。

    Args:
        angle_a: 第一个角度，单位为度。
        angle_b: 第二个角度，单位为度。

    Returns:
        两个角度之间的最小绝对差值。
    """
    return abs(_normalize_angle_deg(angle_a - angle_b))


def _semantic_display_name(semantic_state: str) -> str:
    """将语义状态转换为画面显示文本。

    Args:
        semantic_state: 内部语义状态。

    Returns:
        中文显示文本。
    """
    return SEMANTIC_DISPLAY_NAMES.get(semantic_state, semantic_state)


def _semantic_abbr(semantic_state: str) -> str:
    """将语义状态转换为双字母缩写。

    Args:
        semantic_state: 内部语义状态。

    Returns:
        双字母英文缩写。
    """
    return SEMANTIC_ABBR.get(semantic_state, "UN")


def _semantic_track_policy(semantic_state: str) -> dict[str, float]:
    """获取语义对应的跟踪策略参数。

    Args:
        semantic_state: 内部语义状态。

    Returns:
        语义对应的跟踪策略字典。
    """
    return SEMANTIC_TRACK_POLICY.get(semantic_state, SEMANTIC_TRACK_POLICY["UNKNOWN"])


def _is_static_semantic_state(semantic_state: str) -> bool:
    """判断语义是否属于静止冻结类。

    Args:
        semantic_state: 内部语义状态。

    Returns:
        静止语义返回 True。
    """
    return semantic_state in {"STOP_WAITING", "UNLOADING"}


def _safe_mean(values: list[float]) -> float:
    """计算序列均值。

    Args:
        values: 数值序列。

    Returns:
        序列均值，空序列时返回 0。
    """
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _suppress_overlapping_tracks(
    tracks: list[TrackState], precision_polygon: np.ndarray | None = None
) -> list[TrackState]:
    """抑制同类同区域重叠轨迹的重复显示。

    同一类别中，如果两条轨迹的 IoU 足够高且中心距离足够近，则只保留
    历史更长的那条轨迹用于渲染，另一条仅保留在内部状态中，不显示。

    Args:
        tracks: 待渲染轨迹列表。
        precision_polygon: 精度区域多边形，用于门控语义策略。

    Returns:
        经过冲突抑制后的轨迹列表。
    """
    if len(tracks) <= 1:
        return tracks

    grouped_tracks: dict[int, list[TrackState]] = {}
    for track in tracks:
        grouped_tracks.setdefault(track.cls_id, []).append(track)

    rendered_tracks: list[TrackState] = []
    for cls_id in sorted(grouped_tracks):
        cls_tracks = grouped_tracks[cls_id]
        # 优先保留历史更长的轨迹；同历史时优先保留真实检测框。
        cls_tracks.sort(
            key=lambda track: (
                -track.age,
                -track.hits,
                track.is_predicted,
                -track.conf,
                track.display_id if track.display_id is not None else track.track_id
                if track.track_id is not None else -1,
            )
        )

        kept_tracks: list[TrackState] = []
        for candidate in cls_tracks:
            should_suppress = False
            candidate_policy = _semantic_track_policy(_gated_semantic_state(candidate, precision_polygon))
            for kept_track in kept_tracks:
                kept_policy = _semantic_track_policy(_gated_semantic_state(kept_track, precision_polygon))
                iou_threshold = max(
                    float(candidate_policy["suppress_iou"]),
                    float(kept_policy["suppress_iou"]),
                )
                center_ratio = max(
                    float(candidate_policy["suppress_center_ratio"]),
                    float(kept_policy["suppress_center_ratio"]),
                )

                if _bbox_iou(candidate.bbox, kept_track.bbox) <= iou_threshold:
                    continue

                candidate_diag = _bbox_diagonal(candidate.bbox)
                kept_diag = _bbox_diagonal(kept_track.bbox)
                avg_diag = max(1.0, 0.5 * (candidate_diag + kept_diag))
                center_distance = _bbox_center_distance(candidate.bbox, kept_track.bbox)
                if center_distance <= center_ratio * avg_diag:
                    should_suppress = True
                    break

            if not should_suppress:
                kept_tracks.append(candidate)

        rendered_tracks.extend(kept_tracks)

    return rendered_tracks


class KeyframeBoTSORTTracker:
    """关键帧 BoT-SORT 轨迹管理器。

    该管理器只在关键帧上接收 BoT-SORT 输出，在非关键帧上做运动外推
    做展示补位，适合工程上做“隔 N 帧检测 + 连续播放”的折中方案。

    Args:
        max_missing: 允许连续丢失的最大帧数。
        max_predict_render: 连续预测框允许渲染的最大帧数。
        precision_polygon: 精度区域多边形，None 时全域视为精度区域。
        non_road_polygons: 非道路区域多边形列表。
        handling_polygons: 装卸货区域多边形列表。
        entrance_polygons: 入口区域多边形列表。
    """

    def __init__(
        self,
        max_missing: int,
        max_predict_render: int,
        precision_polygon: np.ndarray | None = None,
        non_road_polygons: list[np.ndarray] | None = None,
        handling_polygons: list[np.ndarray] | None = None,
        entrance_polygons: list[np.ndarray] | None = None,
        reid_extractor: OSNetFeatureExtractor | None = None,
    ):
        self.max_missing = max_missing
        self.max_predict_render = max_predict_render
        self.precision_polygon = precision_polygon
        self.non_road_polygons = non_road_polygons if non_road_polygons is not None else []
        self.handling_polygons = handling_polygons if handling_polygons is not None else []
        self.entrance_polygons = entrance_polygons if entrance_polygons is not None else []
        self.reid_extractor = reid_extractor
        self.frame_count = 0
        self.tracks: dict[int, TrackState] = {}
        self.ghost_tracks: dict[int, TrackState] = {}

    @staticmethod
    def _norm2(x: float, y: float) -> float:
        """计算二维向量模长。

        Args:
            x: 向量 x 分量。
            y: 向量 y 分量。

        Returns:
            向量模长。
        """
        return float(np.hypot(x, y))

    def predict_tracks(self, frame_shape: tuple[int, int, int]) -> None:
        """对所有轨迹执行一次外推预测。

        Args:
            frame_shape: 当前图像尺寸。
        """
        self.frame_count += 1
        frame_h, frame_w = frame_shape[:2]
        for track in self.tracks.values():
            policy = _semantic_track_policy(_gated_semantic_state(track, self.precision_polygon))
            if track.motion_state == "STOPPED":
                # 静止状态下保持框位置不动。
                dx = 0.0
                dy = 0.0
                track.vx = 0.0
                track.vy = 0.0
            else:
                # 根据语义状态调节外推幅度，避免转弯和装卸阶段过冲。
                predict_scale = float(policy["predict_scale"])
                dx = track.vx * predict_scale
                dy = track.vy * predict_scale
                if predict_scale <= 0.0:
                    track.vx = 0.0
                    track.vy = 0.0

            track.bbox = track.bbox + np.array([dx, dy, dx, dy], dtype=np.float32)
            track.bbox = _clip_bbox(track.bbox, frame_w, frame_h)
            track.age += 1
            track.missing_frames += 1
            track.is_predicted = True

            # 静止冻结目标与其他目标重叠时，限制missing_frames防止因遮挡被提前清理
            if track.static_lock and track.missing_frames > 15 and track.anchor_bbox is not None:
                for other in self.tracks.values():
                    if other.track_id == track.track_id:
                        continue
                    if _bbox_iou(track.anchor_bbox, other.bbox) > 0.2:
                        track.missing_frames = 15
                        break

        for ghost_track in self.ghost_tracks.values():
            ghost_track.ghost_age += 1
        self._cleanup_ghost_tracks()

    def _record_semantic_observation(
        self,
        track: TrackState,
        bbox: np.ndarray,
        dx: float,
        dy: float,
    ) -> None:
        """记录单帧观测特征，用于语义状态推断。

        Args:
            track: 当前轨迹状态。
            bbox: 当前目标框。
            dx: 框中心点横向位移。
            dy: 框中心点纵向位移。
        """
        track.center_history.append(_bbox_center(bbox))
        track.speed_history.append(self._norm2(dx, dy))
        track.area_history.append(_bbox_area(bbox))

        move = self._norm2(dx, dy)
        if move >= 0.5:
            heading = float(np.degrees(np.arctan2(dy, dx)))
            track.heading_history.append(_normalize_angle_deg(heading))

    def _infer_semantic_candidate(self, track: TrackState) -> tuple[str, float]:
        """根据轨迹历史特征推断语义候选状态。

        Args:
            track: 当前轨迹状态。

        Returns:
            语义候选状态及其置信度。
        """
        if len(track.center_history) < 2:
            return "UNKNOWN", 0.0

        recent_speeds = list(track.speed_history)[-3:]
        mean_speed = _safe_mean(recent_speeds)
        current_speed = self._norm2(track.vx, track.vy)
        effective_speed = max(mean_speed, current_speed)

        if track.motion_state == "STOPPED":
            if track.stop_count >= SEMANTIC_UNLOADING_CONFIRM_FRAMES:
                area_values = list(track.area_history)[-4:]
                area_stability = 0.0
                if len(area_values) >= 2:
                    area_mean = float(np.mean(area_values))
                    area_std = float(np.std(area_values))
                    if area_mean > 1e-6:
                        area_stability = max(0.0, 1.0 - area_std / area_mean)
                if area_stability >= 1.0 - SEMANTIC_AREA_STABILITY_THRESH:
                    if _is_bbox_in_handling_zone(track.bbox, self.handling_polygons):
                        confidence = min(1.0, track.stop_count / SEMANTIC_UNLOADING_CONFIRM_FRAMES)
                        return "UNLOADING", confidence

            if track.stop_count >= SEMANTIC_WAIT_CONFIRM_FRAMES:
                confidence = min(1.0, track.stop_count / SEMANTIC_WAIT_CONFIRM_FRAMES)
                return "STOP_WAITING", confidence
            return "UNKNOWN", 0.2

        if len(track.heading_history) >= 2 and effective_speed >= SEMANTIC_TURN_MIN_SPEED:
            latest_heading = track.heading_history[-1]
            prev_heading = track.heading_history[-2]
            turn_delta = _angle_diff_deg(latest_heading, prev_heading)
            if turn_delta >= SEMANTIC_TURN_ANGLE_THRESH_DEG:
                confidence = min(1.0, turn_delta / (SEMANTIC_TURN_ANGLE_THRESH_DEG * 2.0))
                return "TURNING", confidence

        if effective_speed >= SEMANTIC_NORMAL_MIN_SPEED:
            confidence = min(1.0, effective_speed / (SEMANTIC_NORMAL_MIN_SPEED * 2.0))
            return "NORMAL_DRIVING", confidence

        return "UNKNOWN", 0.3

    def _refresh_semantic_state(self, track: TrackState) -> None:
        """使用候选状态更新稳定语义状态。

        Args:
            track: 当前轨迹状态。
        """
        candidate_state, candidate_conf = self._infer_semantic_candidate(track)
        if candidate_state == track.semantic_candidate:
            track.semantic_candidate_count += 1
        else:
            track.semantic_candidate = candidate_state
            track.semantic_candidate_count = 1

        if candidate_state == "UNKNOWN":
            track.semantic_state = candidate_state
            track.semantic_conf = candidate_conf
            return

        if candidate_state == track.semantic_state or track.semantic_candidate_count >= SEMANTIC_CONFIRM_FRAMES:
            track.semantic_state = candidate_state
            track.semantic_conf = candidate_conf

    def _lock_static_track(self, track: TrackState) -> None:
        """将轨迹锁定为静止冻结状态。

        Args:
            track: 当前轨迹状态。
        """
        if track.anchor_bbox is None:
            track.anchor_bbox = track.bbox.copy()
        track.static_lock = True
        track.static_mismatch_count = 0

    def _release_static_track(self, track: TrackState) -> None:
        """释放轨迹的静止冻结状态。

        Args:
            track: 当前轨迹状态。
        """
        track.static_lock = False
        track.static_mismatch_count = 0
        track.anchor_bbox = None

    def _is_static_observation(self, track: TrackState, bbox: np.ndarray) -> bool:
        """判断观测框是否仍然支持静止冻结状态。

        Args:
            track: 当前轨迹状态。
            bbox: 当前观测框。

        Returns:
            与静止锚点一致时返回 True。
        """
        reference_bbox = track.anchor_bbox if track.anchor_bbox is not None else track.bbox
        iou = _bbox_iou(reference_bbox, bbox)
        center_distance = _bbox_center_distance(reference_bbox, bbox)
        reference_diag = max(1.0, _bbox_diagonal(reference_bbox))
        return iou >= STATIC_LOCK_IOU and center_distance <= STATIC_LOCK_CENTER_RATIO * reference_diag

    # -- 多线索联合匹配（IoU + 外观） --

    def _reid_multi_cue_score(
        self,
        query_bbox: np.ndarray,
        ref_bbox: np.ndarray,
        query_feat: np.ndarray | None,
        track: TrackState,
    ) -> tuple[float, float, float]:
        """计算 IoU + 外观特征的多线索联合匹配得分。

        加权融合公式:
            score = α × iou_norm + (1−α) × cos_sim
        其中 α 随 track.missing_frames 线性衰减。

        当 reid_extractor 不可用或 query_feat 为 None 时，α 固定为 1.0（纯 IoU 模式）。
        """
        def _reid_iou_thresh(missing: int) -> float:
            decay = min(1.0, missing / REID_MISSING_DECAY_FRAMES)
            return max(REID_IOU_MIN, REID_IOU_THRESH * (1.0 - decay))

        iou_val = _bbox_iou(query_bbox, ref_bbox)
        iou_thresh = _reid_iou_thresh(track.missing_frames)
        iou_norm = iou_val / max(iou_thresh, 1e-6)

        # 外观匹配
        cos_sim = 0.0
        use_appearance = (
            self.reid_extractor is not None
            and query_feat is not None
            and len(track.appearance_features) > 0
        )
        if use_appearance:
            cos_sim = _cosine_similarity_max(query_feat, track.appearance_features)

        # 融合权重
        if use_appearance and len(track.appearance_features) > 0:
            alpha_range = REID_APPEARANCE_WEIGHT_INIT - REID_APPEARANCE_WEIGHT_MIN
            decay_ratio = min(1.0, track.missing_frames / REID_APPEARANCE_DECAY_FRAMES)
            alpha = max(
                REID_APPEARANCE_WEIGHT_MIN,
                REID_APPEARANCE_WEIGHT_INIT - decay_ratio * alpha_range,
            )
        else:
            alpha = 1.0

        fusion = alpha * iou_norm + (1.0 - alpha) * cos_sim
        return fusion, iou_norm, cos_sim

    def _is_reid_match(
        self,
        query_bbox: np.ndarray,
        ref_bbox: np.ndarray,
        query_feat: np.ndarray | None,
        track: TrackState,
    ) -> bool:
        """判断检测框是否与轨迹匹配（多线索联合判定）。"""
        fusion, iou_score, cos_sim = self._reid_multi_cue_score(
            query_bbox, ref_bbox, query_feat, track,
        )

        if fusion < 0.5:
            return False

        if self.reid_extractor is not None and query_feat is not None and len(track.appearance_features) > 0:
            if cos_sim < REID_COSINE_THRESH:
                return False

        return True

    def _store_appearance_feature(
        self, track: TrackState, feature: np.ndarray | None
    ) -> None:
        """将外观特征写入轨迹历史。

        运动目标使用 EMA 更新，保持特征平滑；静止目标直接存储。
        """
        if feature is None:
            return
        if track.last_appearance_feature is not None and track.motion_state == "MOVE":
            alpha = REID_APPEARANCE_EMA_ALPHA
            blended = alpha * feature + (1.0 - alpha) * track.last_appearance_feature
            norm = float(np.linalg.norm(blended))
            if norm > 1e-8:
                blended = blended / norm
            feature = blended
        track.appearance_features.append(feature)
        track.last_appearance_feature = feature

    def _reid_ref_bbox(self, track: TrackState) -> np.ndarray:
        """获取 ReID 匹配的参考框。

        静止目标用冻结锚点框匹配（不受漂移影响），
        运动目标用当前预测框匹配。
        """
        if _is_static_semantic_state(track.semantic_state) and track.anchor_bbox is not None:
            return track.anchor_bbox
        return track.bbox

    # -- 主更新逻辑 --

    def update_from_result(self, result, frame_shape: tuple[int, int, int], frame: np.ndarray | None = None) -> None:
        """使用关键帧 BoT-SORT 结果回写轨迹状态。

        Args:
            result: YOLO track 的单帧结果。
            frame_shape: 当前图像尺寸。
            frame: 原始 BGR 帧图像，用于 OSNet 特征提取。None 时跳过外观提取（回退为纯 IoU 模式）。

        说明：
            当检测框置信度低于展示阈值时，只用它更新轨迹速度与关联状态，
            但保持上一帧的预测框继续显示，避免低置信度框直接出现在画面上。
        """
        if result.boxes is None or len(result.boxes) == 0 or result.boxes.id is None:
            self._remove_expired_tracks()
            return

        frame_h, frame_w = frame_shape[:2]
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        # 批量提取外观特征
        appearance_features: dict[int, np.ndarray | None] = {}
        if self.reid_extractor is not None and frame is not None:
            bboxes_for_reid = []
            bbox_indices = []
            for idx, (bbox, cls_id, conf, track_id_) in enumerate(zip(boxes_xyxy, classes, confs, track_ids)):
                if track_id_ < 0:
                    continue
                clipped = _clip_bbox(bbox.astype(np.float32), frame_w, frame_h)
                if self.non_road_polygons and _is_bbox_in_non_road(clipped, self.non_road_polygons):
                    continue
                if float(conf) >= CONF_THRES_DISPLAY:
                    bboxes_for_reid.append(clipped)
                else:
                    bboxes_for_reid.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
                bbox_indices.append(idx)

            valid_bboxes = [b for b in bboxes_for_reid if _bbox_area(b) > 0]
            if valid_bboxes:
                valid_features = self.reid_extractor.extract_features_batch(frame, valid_bboxes)

                feat_idx = 0
                for i, bbox in enumerate(bboxes_for_reid):
                    if _bbox_area(bbox) > 0 and feat_idx < len(valid_features):
                        appearance_features[int(track_ids[bbox_indices[i]])] = valid_features[feat_idx]
                        feat_idx += 1
                    else:
                        appearance_features[int(track_ids[bbox_indices[i]])] = None

        for bbox, cls_id, conf, track_id in zip(boxes_xyxy, classes, confs, track_ids):
            if track_id < 0:
                continue

            bbox = _clip_bbox(bbox.astype(np.float32), frame_w, frame_h)
            cls_name = result.names.get(int(cls_id), str(int(cls_id)))
            if self.non_road_polygons and _is_bbox_in_non_road(bbox, self.non_road_polygons):
                continue
            is_displayable = float(conf) >= CONF_THRES_DISPLAY

            # 获取当前检测的外观特征
            current_feat = appearance_features.get(int(track_id), None)
            if track_id not in self.tracks:
                # 重识别：IoU + 外观特征联合匹配丢失轨迹
                reid_target = None
                reid_source = None  # "track" 或 "ghost"

                # 先查活跃轨迹中的丢失轨迹
                for existing_id, existing_track in self.tracks.items():
                    if existing_track.missing_frames < 15:
                        continue
                    if existing_track.cls_id != int(cls_id):
                        continue
                    ref_bbox = self._reid_ref_bbox(existing_track)
                    if self._is_reid_match(
                        bbox, ref_bbox, current_feat, existing_track,
                    ):
                        reid_target = (existing_id, existing_track)
                        reid_source = "track"
                        break

                # 再查幽灵轨迹缓存
                if reid_target is None:
                    for ghost_key, ghost_track in self.ghost_tracks.items():
                        if ghost_track.cls_id != int(cls_id):
                            continue
                        ref_bbox = self._reid_ref_bbox(ghost_track)
                        if self._is_reid_match(
                            bbox, ref_bbox, current_feat, ghost_track,
                        ):
                            reid_target = (ghost_key, ghost_track)
                            reid_source = "ghost"
                            break

                if reid_target is not None:
                    old_key, old_track = reid_target
                    if old_track.display_id is None:
                        old_track.display_id = old_key
                    old_track.track_id = track_id
                    old_track.bbox = bbox
                    old_track.last_observed_bbox = bbox.copy()
                    old_track.cls_id = int(cls_id)
                    old_track.cls_name = cls_name
                    old_track.conf = float(conf)
                    old_track.missing_frames = 0
                    old_track.is_predicted = False
                    old_track.hits += 1
                    old_track.age += 1
                    old_track.ghost_age = 0
                    # 存储当前检测的外观特征
                    self._store_appearance_feature(old_track, current_feat)
                    if reid_source == "track":
                        del self.tracks[old_key]
                    else:
                        del self.ghost_tracks[old_key]
                    self.tracks[track_id] = old_track
                    self._record_semantic_observation(
                        track=old_track,
                        bbox=bbox,
                        dx=0.0,
                        dy=0.0,
                    )
                    self._refresh_semantic_state(old_track)
                    continue

                if (
                    self.frame_count > NEW_TRACK_ENTRANCE_GATE_FRAMES
                    and self.entrance_polygons
                    and self.precision_polygon is not None
                    and _is_bbox_in_precision_zone(bbox, self.precision_polygon)
                    and not _is_bbox_in_entrance_zone(bbox, self.entrance_polygons)
                ):
                    continue
                new_track = TrackState(
                    track_id=track_id,
                    cls_id=int(cls_id),
                    cls_name=cls_name,
                    bbox=bbox,
                    conf=float(conf),
                    is_predicted=False,
                    last_observed_bbox=bbox.copy(),
                )
                self._store_appearance_feature(new_track, current_feat)
                self.tracks[track_id] = new_track
                self._record_semantic_observation(
                    track=self.tracks[track_id],
                    bbox=bbox,
                    dx=0.0,
                    dy=0.0,
                )
                self._refresh_semantic_state(self.tracks[track_id])
                continue

            track = self.tracks[track_id]

            # 存储外观特征（即使是已知轨迹也更新，为后续 ReID 做准备）
            self._store_appearance_feature(track, current_feat)

            if track.static_release_cooldown > 0:
                track.static_release_cooldown -= 1
            if track.anchor_bbox is None and _gated_is_static_semantic(track, self.precision_polygon):
                track.anchor_bbox = track.bbox.copy()

            static_protected = track.static_lock or _gated_is_static_semantic(track, self.precision_polygon)
            if static_protected:
                if track.anchor_bbox is None:
                    track.anchor_bbox = track.bbox.copy()

                if track.semantic_state == "UNLOADING":
                    skip_observation = False
                    for other in self.tracks.values():
                        if other.track_id == track_id:
                            continue
                        if other.semantic_state in ("UNLOADING", "STOP_WAITING"):
                            continue
                        if _bbox_iou(track.anchor_bbox, other.bbox) > 0.0:
                            skip_observation = True
                            break
                    if skip_observation:
                        track.bbox = track.anchor_bbox.copy()
                        track.vx = 0.0
                        track.vy = 0.0
                        track.motion_state = "STOPPED"
                        track.age += 1
                        track.missing_frames = 0
                        track.is_predicted = True
                        track.static_lock = True
                        track.stop_count += 1
                        self._record_semantic_observation(track=track, bbox=track.bbox, dx=0.0, dy=0.0)
                        self._refresh_semantic_state(track)
                        continue

                outlier_dist = _bbox_center_distance(track.anchor_bbox, bbox)
                anchor_diag = max(1.0, _bbox_diagonal(track.anchor_bbox))
                if outlier_dist <= STATIC_OUTLIER_DIAG_RATIO * anchor_diag:
                    if self._is_static_observation(track, bbox):
                        track.static_mismatch_count = max(0, track.static_mismatch_count - 1)
                    else:
                        track.static_mismatch_count += 1

                stc_policy = _semantic_track_policy(track.semantic_state)
                static_release_limit = stc_policy.get(
                    "static_lock_release_frames", 3
                )
                if track.static_mismatch_count < static_release_limit:
                    track.bbox = track.anchor_bbox.copy()
                    track.vx = 0.0
                    track.vy = 0.0
                    track.motion_state = "STOPPED"
                    track.conf = float(conf)
                    track.hits += 1
                    track.age += 1
                    track.missing_frames = 0
                    track.is_predicted = True
                    track.static_lock = True
                    track.stop_count += 1
                    self._record_semantic_observation(track=track, bbox=track.bbox, dx=0.0, dy=0.0)
                    self._refresh_semantic_state(track)
                    continue

                self._release_static_track(track)
                track.stop_count = 0
                track.motion_state = "MOVE"
                track.semantic_state = "UNKNOWN"
                track.semantic_conf = 0.0
                track.semantic_candidate = "UNKNOWN"
                track.semantic_candidate_count = 0
                track.static_release_cooldown = STATIC_RELEASE_COOLDOWN

            old_cx, old_cy = _bbox_center(track.bbox)
            new_cx, new_cy = _bbox_center(bbox)
            dx = new_cx - old_cx
            dy = new_cy - old_cy

            # 使用观测位移更新速度，运动阶段只保留匀速模型。
            track.vx = 0.7 * track.vx + 0.3 * dx
            track.vy = 0.7 * track.vy + 0.3 * dy

            if is_displayable:
                track.bbox = bbox
            else:
                pass
            track.last_observed_bbox = bbox.copy()

            speed = self._norm2(track.vx, track.vy)
            move = self._norm2(dx, dy)
            bbox_diag = _bbox_diagonal(track.bbox)
            is_static_now = (
                speed <= STOP_SPEED_DIAG_RATIO * bbox_diag
                and move <= STOP_MOVE_DIAG_RATIO * bbox_diag
            )
            if is_static_now:
                track.stop_count += 1
            else:
                track.stop_count = 0

            if track.stop_count >= STOP_CONFIRM_FRAMES:
                track.motion_state = "STOPPED"
                track.vx = 0.0
                track.vy = 0.0
                if _gated_is_static_semantic(track, self.precision_polygon):
                    self._lock_static_track(track)
                    if track.anchor_bbox is None:
                        track.anchor_bbox = track.bbox.copy()
            else:
                track.motion_state = "MOVE"
                if not _gated_is_static_semantic(track, self.precision_polygon):
                    if track.static_lock:
                        self._release_static_track(track)
                        track.static_release_cooldown = STATIC_RELEASE_COOLDOWN

            track.conf = float(conf)
            track.hits += 1
            track.age += 1
            track.missing_frames = 0
            track.is_predicted = not is_displayable
            self._record_semantic_observation(track=track, bbox=bbox, dx=dx, dy=dy)
            self._refresh_semantic_state(track)

        self._remove_expired_tracks()

    def get_render_tracks(self, min_hits: int, min_display_conf: float) -> list[TrackState]:
        """获取当前应渲染的轨迹列表。

        Args:
            min_hits: 最少命中次数。
            min_display_conf: 低于该置信度的真实检测框不展示。

        Returns:
            可渲染轨迹列表。
        """
        render_tracks: list[TrackState] = []
        for track in self.tracks.values():
            policy = _semantic_track_policy(_gated_semantic_state(track, self.precision_polygon))
            effective_min_display_conf = min_display_conf * float(policy["display_conf_scale"])
            effective_max_missing = max(
                self.max_missing,
                int(round(self.max_missing * float(policy["max_missing_scale"]))),
            )
            effective_max_predict_render = max(
                self.max_predict_render,
                int(round(self.max_predict_render * float(policy["max_predict_render_scale"]))),
            )
            if _gated_is_static_semantic(track, self.precision_polygon) and track.is_predicted:
                effective_max_predict_render = max(effective_max_predict_render, max(1, effective_max_missing - 1))
            if track.hits < min_hits:
                continue
            if track.is_predicted:
                if track.missing_frames <= effective_max_predict_render:
                    render_tracks.append(track)
                continue
            if track.conf >= effective_min_display_conf:
                render_tracks.append(track)
        return _suppress_overlapping_tracks(render_tracks, self.precision_polygon)

    def _remove_expired_tracks(self) -> None:
        """将连续丢失时间过长的轨迹移入幽灵缓存，供后续重识别使用。"""
        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missing_frames
            > int(round(self.max_missing * float(_semantic_track_policy(_gated_semantic_state(track, self.precision_polygon))["max_missing_scale"])))
        ]
        for track_id in expired_ids:
            track = self.tracks[track_id]
            track.ghost_age = 0
            ghost_key = track.display_id if track.display_id is not None else track_id
            self.ghost_tracks[ghost_key] = track
            del self.tracks[track_id]

    def _cleanup_ghost_tracks(self) -> None:
        """清理幽灵缓存中超时的轨迹。"""
        expired_keys = [
            k for k, t in self.ghost_tracks.items()
            if t.ghost_age > GHOST_MAX_AGE
        ]
        for k in expired_keys:
            del self.ghost_tracks[k]


def _extract_tracked_boxes(result) -> list[TrackedBox]:
    """从 YOLO 跟踪结果中提取绘制信息。

    Args:
        result: YOLO 单帧 track 结果。

    Returns:
        跟踪框列表。
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)
    else:
        track_ids = np.full(len(boxes_xyxy), -1, dtype=int)

    tracked_boxes: list[TrackedBox] = []
    for bbox, cls_id, conf, track_id in zip(boxes_xyxy, classes, confs, track_ids):
        cls_name = result.names.get(int(cls_id), str(int(cls_id)))
        tracked_boxes.append(
            TrackedBox(
                bbox=bbox.astype(np.float32),
                cls_id=int(cls_id),
                cls_name=cls_name,
                conf=float(conf),
                track_id=None if track_id < 0 else int(track_id),
            )
        )
    return tracked_boxes


def _draw_tracks(frame: np.ndarray, tracks: list[TrackState]) -> np.ndarray:
    """将 BoT-SORT 跟踪结果绘制到图像上。

    Args:
        frame: 输入图像。
        tracks: 跟踪目标列表。

    Returns:
        绘制后的图像。
    """
    canvas = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.bbox.astype(int).tolist()
        color = (0, 165, 255) if track.is_predicted else _class_color(track.cls_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, LINE_THICKNESS, cv2.LINE_AA)

        if SHOW_LABEL:
            display_id = track.display_id if track.display_id is not None else track.track_id
            if display_id is not None:
                label = f"ID:{display_id} {track.cls_name}"
            else:
                label = track.cls_name

            semantic_abbr = _semantic_abbr(track.semantic_state)
            label = f"{label} | {semantic_abbr}"
            if track.semantic_conf > 0.0:
                label = f"{label} {track.semantic_conf:.2f}"

            if track.is_predicted:
                label = f"{label} PRED"
            elif SHOW_CONF and track.conf >= CONF_THRES_DISPLAY:
                label = f"{label} {track.conf:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_text = max(0, y1 - th - 8)
            cv2.rectangle(canvas, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, y_text + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def _init_reid_extractor() -> OSNetFeatureExtractor | None:
    """初始化 OSNet 特征提取器。

    通过 torchreid 构建 OSNet 架构，并从 REID_MODEL_PATH 加载预训练权重。

    Returns:
        OSNetFeatureExtractor 实例，torchreid 不可用时返回 None。
    """
    try:
        import torchreid  # noqa: F401
    except ImportError:
        print("torchreid 未安装，OSNet ReID 不可用")
        return None
    device = DEVICE if isinstance(DEVICE, str) else f"cuda:{DEVICE}"
    return OSNetFeatureExtractor(
        model_path=REID_MODEL_PATH,
        device=device,
        use_half=USE_HALF,
        image_size=REID_IMAGE_SIZE,
    )


def run_detection() -> None:
    """执行 N 帧关键帧 BoT-SORT 跟踪主流程。

    说明：
        该模式在关键帧上执行检测与关联，在非关键帧上对轨迹做匀速外推，
        用于平衡播放流畅度、推理速度和轨迹连续性。
    """
    model_path = str(MODEL_PATH)
    video_path = str(VIDEO_PATH)
    tracker_cfg_path = str(TRACKER_CFG_PATH)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not Path(tracker_cfg_path).exists():
        raise FileNotFoundError(f"跟踪器配置文件不存在: {tracker_cfg_path}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频加载成功: 共 {total_frames} 帧 | FPS: {fps}")
    print(
        f"跟踪策略: BoT-SORT + OSNet ReID | det_interval={DET_INTERVAL} | conf={CONF_THRES_TRACK} | "
        f"display_conf={CONF_THRES_DISPLAY} | imgsz={IMGSZ} | tracker={TRACKER_CFG_PATH.name}"
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    current_frame_idx = START_FRAME
    first_debug_printed = False
    names: dict[int, str] = {}
    precision_polygon = _load_precision_zone_polygon(Path(video_path))
    if precision_polygon is not None:
        print(f"已加载精度区域多边形: {len(precision_polygon)} 个顶点")
    else:
        print("未找到精度区域配置，全域启用语义辅助追踪")

    non_road_polygons = _load_non_road_polygons(Path(video_path))
    if non_road_polygons:
        print(f"已加载 {len(non_road_polygons)} 个非道路区域多边形")

    handling_polygons = _load_handling_polygons(Path(video_path))
    if handling_polygons:
        print(f"已加载 {len(handling_polygons)} 个装卸货区域多边形")

    entrance_polygons = _load_entrance_polygons(Path(video_path))
    if entrance_polygons:
        print(f"已加载 {len(entrance_polygons)} 个入口区域多边形")

    # 初始化 OSNet 特征提取器
    reid_extractor = _init_reid_extractor()
    if reid_extractor is not None:
        print("OSNet 外观特征提取器已就绪，启用 IoU + 外观联合 ReID 匹配")
    else:
        print("OSNet 不可用，回退为纯 IoU ReID 匹配模式")

    tracker = KeyframeBoTSORTTracker(
        max_missing=TRACK_MAX_MISSING,
        max_predict_render=TRACK_PREDICT_RENDER,
        precision_polygon=precision_polygon,
        non_road_polygons=non_road_polygons,
        handling_polygons=handling_polygons,
        entrance_polygons=entrance_polygons,
        reid_extractor=reid_extractor,
    )

    if DISPLAY_MODE == "opencv":
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n视频播放结束或读取失败")
                break

            tracker.predict_tracks(frame.shape)
            should_detect = (current_frame_idx - START_FRAME) % DET_INTERVAL == 0
            if should_detect:
                results = model.track(
                    frame,
                    agnostic_nms=False,  # 同类严格去重
                    imgsz=IMGSZ,
                    conf=CONF_THRES_TRACK,
                    iou=NMS_IOU,
                    verbose=False,
                    persist=True,
                    tracker=tracker_cfg_path,
                    device=DEVICE,
                    half=USE_HALF,
                )
                result = results[0]
                names = result.names
                tracker.update_from_result(result, frame.shape, frame=frame)

            tracks = tracker.get_render_tracks(
                min_hits=TRACK_MIN_HITS,
                min_display_conf=CONF_THRES_DISPLAY,
            )
            annotated_frame = _draw_tracks(frame, tracks)
            display_frame = annotated_frame

            if SHOW_WHITEBOARD:
                boxes_xyxy = [track.bbox.tolist() for track in tracks]
                classes = [track.cls_id for track in tracks]
                semantic_states = [track.semantic_state for track in tracks]
                whiteboard = _draw_whiteboard(
                    frame_shape=annotated_frame.shape,
                    boxes_xyxy=boxes_xyxy,
                    classes=classes,
                    names=names,
                    semantic_states=semantic_states,
                    precision_polygon=precision_polygon,
                )
                display_frame = cv2.hconcat([annotated_frame, whiteboard])
                split_x = annotated_frame.shape[1]
                cv2.line(display_frame, (split_x, 0), (split_x, display_frame.shape[0] - 1), (0, 0, 0), 2)

                if not first_debug_printed:
                    print(
                        f"拼接成功: left={annotated_frame.shape[1]}x{annotated_frame.shape[0]}, "
                        f"right={whiteboard.shape[1]}x{whiteboard.shape[0]}, "
                        f"merged={display_frame.shape[1]}x{display_frame.shape[0]}"
                    )
                    first_debug_printed = True

            status_text = (
                f"Frame:{current_frame_idx}/{total_frames} "
                f"Det:{'Y' if should_detect else 'N'} "
                f"Interval:{DET_INTERVAL} "
                f"Tracks:{len(tracks)} "
                f"BoT-SORT{' + OSNet' if reid_extractor is not None else ''}"
            )
            cv2.putText(
                display_frame,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

            if DISPLAY_MODE == "opencv":
                if AUTO_FIT_SCREEN:
                    screen_w, screen_h = _get_screen_size()
                    if screen_w is not None and screen_h is not None:
                        avail_w = max(1, screen_w - SCREEN_MARGIN)
                        avail_h = max(1, screen_h - SCREEN_MARGIN)
                        display_show = _fit_with_letterbox(display_frame, avail_w, avail_h)
                    else:
                        if DISPLAY_SCALE != 1.0:
                            display_show = cv2.resize(
                                display_frame,
                                None,
                                fx=DISPLAY_SCALE,
                                fy=DISPLAY_SCALE,
                                interpolation=cv2.INTER_AREA,
                            )
                        else:
                            display_show = display_frame
                else:
                    if DISPLAY_SCALE != 1.0:
                        display_show = cv2.resize(
                            display_frame,
                            None,
                            fx=DISPLAY_SCALE,
                            fy=DISPLAY_SCALE,
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        display_show = display_frame

                cv2.imshow("Detection", display_show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n手动停止播放")
                    break
            else:
                img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                if clear_output is not None:
                    clear_output(wait=True)
                plt.figure(figsize=(14, 8))
                plt.imshow(img_rgb)
                plt.axis("off")
                plt.title(
                    f"Frame: {current_frame_idx}/{total_frames} | BoT-SORT | DET_INTERVAL={DET_INTERVAL}",
                    fontsize=14,
                )
                plt.show()

            if INTERVAL_SECONDS > 0:
                time.sleep(INTERVAL_SECONDS)

            current_frame_idx += 1

    except KeyboardInterrupt:
        print("\n已手动停止播放")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close("all")


if __name__ == "__main__":
    run_detection()
