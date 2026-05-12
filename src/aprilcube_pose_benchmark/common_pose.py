from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import aprilcube
import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parents[2]
PROJECT_ROOT = THIS_FILE.parents[4]
RECORDER_UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"
if str(RECORDER_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(RECORDER_UTILS_DIR))

from april_tag_detector import TemporalTagPoseEstimator  # noqa: E402


PUPIL_TO_CUBE_CORNER_INDEX = [2, 1, 0, 3]


def apriltag_family_from_dict_name(dict_name: str) -> str:
    family_map = {
        "apriltag_16h5": "tag16h5",
        "apriltag_25h9": "tag25h9",
        "apriltag_36h10": "tag36h10",
        "apriltag_36h11": "tag36h11",
    }
    if dict_name not in family_map:
        raise ValueError(f"Unsupported AprilTag family in cube config: {dict_name}")
    return family_map[dict_name]


def camera_matrix_to_intrinsic_dict(k: np.ndarray) -> dict[str, float]:
    k = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def k_to_camera_params(k: np.ndarray) -> tuple[float, float, float, float]:
    k = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def build_detector_from_record(
    cube_path: str | Path,
    camera_record: dict[str, Any],
) -> Any:
    camera_matrix = np.asarray(camera_record["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(camera_record.get("dist_coeffs", np.zeros(5)), dtype=np.float64).reshape(-1)
    return aprilcube.detector(
        Path(cube_path).expanduser().resolve(),
        intrinsic_cfg=camera_matrix_to_intrinsic_dict(camera_matrix),
        dist_coeffs=dist_coeffs,
        enable_filter=False,
        fast=True,
    )


def create_native_detector(detector: Any) -> Detector:
    family = apriltag_family_from_dict_name(detector.config.dict_name)
    return Detector(families=family, quad_decimate=1.0)


def detect_pupil_tags(
    detector: Any,
    native_detector: Detector,
    gray: np.ndarray,
    *,
    estimate_tag_pose: bool,
) -> list[Any]:
    kwargs: dict[str, Any] = {"estimate_tag_pose": bool(estimate_tag_pose)}
    if estimate_tag_pose:
        kwargs["camera_params"] = k_to_camera_params(detector.camera_matrix)
        kwargs["tag_size"] = float(detector.config.tag_size_mm) / 1000.0
    tags = native_detector.detect(np.asarray(gray, dtype=np.uint8), **kwargs)
    return [tag for tag in tags if int(tag.tag_id) in detector.valid_ids]


def is_valid_rotation_matrix(rot: np.ndarray, det_tol: float = 0.2, ortho_tol: float = 0.2) -> bool:
    rot = np.asarray(rot, dtype=np.float64)
    if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
        return False
    det = float(np.linalg.det(rot))
    if det <= 0.0 or abs(det - 1.0) > det_tol:
        return False
    return float(np.linalg.norm(rot.T @ rot - np.eye(3))) <= ortho_tol


def empty_result() -> dict[str, Any]:
    return {
        "success": False,
        "rvec": None,
        "tvec": None,
        "T": None,
        "reproj_error": float("inf"),
        "n_tags": 0,
        "n_inliers": 0,
        "detections": [],
        "tag_ids": [],
        "visible_faces": set(),
        "predicted": False,
        "tag_pose_by_id": {},
        "algorithm_debug": {},
    }


def visible_faces_for_ids(detector: Any, tag_ids: list[int]) -> set[str]:
    faces: set[str] = set()
    for tag_id in tag_ids:
        for face_name, id_set in detector.face_id_sets.items():
            if int(tag_id) in id_set:
                faces.add(face_name)
    return faces


def detections_from_tags(tags: list[Any]) -> list[tuple[int, np.ndarray]]:
    return [(int(tag.tag_id), np.asarray(tag.corners, dtype=np.float64).reshape(4, 2)) for tag in tags]


def reorder_pupil_corners_to_cube_order(corners: np.ndarray) -> np.ndarray:
    arr = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return arr[np.asarray(PUPIL_TO_CUBE_CORNER_INDEX, dtype=np.int64)]


def aggregate_cube_correspondences(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    object_points = []
    image_points = []
    for tag_id, corners in detections:
        corners_3d = detector.tag_corner_map.get(int(tag_id), None)
        if corners_3d is None:
            continue
        object_points.append(np.asarray(corners_3d, dtype=np.float64).reshape(4, 3))
        image_points.append(reorder_pupil_corners_to_cube_order(corners))
    if not object_points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.vstack(object_points), np.vstack(image_points)


def tag_object_corners(tag_size_m: float) -> np.ndarray:
    half = float(tag_size_m) * 0.5
    return np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float64,
    )


def pnp_cube_pose_lm(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    *,
    use_ransac: bool,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    object_points, image_points = aggregate_cube_correspondences(detector, detections)
    if len(object_points) < 4:
        return None

    k = np.asarray(detector.camera_matrix, dtype=np.float64)
    dist = np.asarray(detector.dist_coeffs, dtype=np.float64).reshape(-1, 1)
    inlier_count = len(object_points)

    if use_ransac and len(object_points) >= 8:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            k,
            dist,
            iterationsCount=100,
            reprojectionError=4.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        if inliers is not None and len(inliers) >= 4:
            inlier_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
            object_points_ref = object_points[inlier_idx]
            image_points_ref = image_points[inlier_idx]
            inlier_count = len(inlier_idx)
        else:
            object_points_ref = object_points
            image_points_ref = image_points
    else:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            k,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        object_points_ref = object_points
        image_points_ref = image_points

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points_ref,
            image_points_ref,
            k,
            dist,
            rvec,
            tvec,
        )
    except cv2.error:
        pass

    reproj = reprojection_error(detector, detections, rvec, tvec)
    return rvec.reshape(3, 1), tvec.reshape(3, 1), reproj, inlier_count


def reprojection_error(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    object_points, image_points = aggregate_cube_correspondences(detector, detections)
    if len(object_points) == 0:
        return float("inf")
    projected, _ = cv2.projectPoints(
        object_points,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        detector.camera_matrix,
        detector.dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - image_points, axis=1)))


def result_from_pose(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    rvec: np.ndarray,
    tvec: np.ndarray,
    *,
    reproj_error: float,
    n_inliers: int,
    debug: dict[str, Any] | None = None,
    tag_pose_by_id: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = empty_result()
    tag_ids = [int(tag_id) for tag_id, _ in detections]
    result.update(
        {
            "success": True,
            "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            "reproj_error": float(reproj_error),
            "n_tags": len(detections),
            "n_inliers": int(n_inliers),
            "detections": detections,
            "tag_ids": tag_ids,
            "visible_faces": visible_faces_for_ids(detector, tag_ids),
            "tag_pose_by_id": tag_pose_by_id or {},
            "algorithm_debug": debug or {},
        }
    )
    transform = np.eye(4, dtype=np.float64)
    rot, _ = cv2.Rodrigues(result["rvec"])
    transform[:3, :3] = rot
    transform[:3, 3] = result["tvec"].reshape(3)
    result["T"] = transform
    return result


def tag_pose_from_native_detection(tag: Any) -> tuple[np.ndarray, np.ndarray, float | None] | None:
    pose_R = getattr(tag, "pose_R", None)
    pose_t = getattr(tag, "pose_t", None)
    if pose_R is None or pose_t is None:
        return None
    pose_R = np.asarray(pose_R, dtype=np.float64).reshape(3, 3)
    if not is_valid_rotation_matrix(pose_R):
        return None
    pose_t_mm = np.asarray(pose_t, dtype=np.float64).reshape(3, 1) * 1000.0
    pose_err = getattr(tag, "pose_err", None)
    return pose_R, pose_t_mm, (float(pose_err) if pose_err is not None else None)


def build_tag_to_cube_transform(detector: Any, tag_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    corners_3d = detector.tag_corner_map.get(int(tag_id), None)
    if corners_3d is None:
        return None
    tl, tr, _br, bl = np.asarray(corners_3d, dtype=np.float64).reshape(4, 3)
    x_axis = tr - tl
    y_axis = bl - tl
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rot_cube_tag = np.column_stack((x_axis, y_axis, z_axis))
    if not is_valid_rotation_matrix(rot_cube_tag):
        return None
    center_cube = np.mean(np.asarray(corners_3d, dtype=np.float64).reshape(4, 3), axis=0).reshape(3, 1)
    return rot_cube_tag, center_cube


def cube_pose_from_tag_pose(
    detector: Any,
    tag_id: int,
    tag_rot_mat: np.ndarray,
    tag_tvec_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    tag_to_cube = build_tag_to_cube_transform(detector, tag_id)
    if tag_to_cube is None:
        return None
    rot_cube_tag, center_cube = tag_to_cube
    rot_tag_cube = rot_cube_tag.T
    center_tag = -rot_tag_cube @ center_cube
    rot_cam_tag = np.asarray(tag_rot_mat, dtype=np.float64).reshape(3, 3)
    tag_tvec_mm = np.asarray(tag_tvec_mm, dtype=np.float64).reshape(3, 1)
    rot_cam_cube = rot_cam_tag @ rot_tag_cube
    center_cam = rot_cam_tag @ center_tag + tag_tvec_mm
    if not is_valid_rotation_matrix(rot_cam_cube):
        return None
    return rot_cam_cube, center_cam


def average_rotations(rot_mats: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray | None:
    if not rot_mats:
        return None
    if weights is None:
        weights = np.ones(len(rot_mats), dtype=np.float64) / len(rot_mats)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    accum = np.zeros((3, 3), dtype=np.float64)
    for rot, weight in zip(rot_mats, weights):
        accum += float(weight) * np.asarray(rot, dtype=np.float64).reshape(3, 3)
    u, _s, vt = np.linalg.svd(accum)
    rot_avg = u @ vt
    if np.linalg.det(rot_avg) < 0.0:
        u[:, -1] *= -1.0
        rot_avg = u @ vt
    return rot_avg if is_valid_rotation_matrix(rot_avg) else None


def fuse_cube_candidates(candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    if not candidates:
        return None
    weights = []
    rot_mats = []
    tvecs = []
    for cand in candidates:
        err = cand.get("reproj_error", None)
        weight = 1.0 if err is None else 1.0 / max(float(err), 1e-3)
        weights.append(weight)
        rot_mats.append(np.asarray(cand["rot_mat"], dtype=np.float64).reshape(3, 3))
        tvecs.append(np.asarray(cand["tvec"], dtype=np.float64).reshape(3, 1))
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr /= max(float(np.sum(weights_arr)), 1e-12)
    rot_avg = average_rotations(rot_mats, weights_arr)
    if rot_avg is None:
        return None
    t_avg = np.zeros((3, 1), dtype=np.float64)
    for weight, tvec in zip(weights_arr, tvecs):
        t_avg += float(weight) * tvec
    return rot_avg, t_avg


def result_from_cube_candidates(
    detector: Any,
    detections: list[tuple[int, np.ndarray]],
    candidates: list[dict[str, Any]],
    *,
    debug: dict[str, Any] | None = None,
    tag_pose_by_id: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fused = fuse_cube_candidates(candidates)
    if fused is None:
        result = empty_result()
        result["detections"] = detections
        result["n_tags"] = len(detections)
        result["tag_ids"] = [int(tag_id) for tag_id, _ in detections]
        result["visible_faces"] = visible_faces_for_ids(detector, result["tag_ids"])
        result["algorithm_debug"] = debug or {}
        return result
    rot_mat, tvec = fused
    rvec, _ = cv2.Rodrigues(rot_mat)
    reproj = reprojection_error(detector, detections, rvec, tvec)
    return result_from_pose(
        detector,
        detections,
        rvec,
        tvec,
        reproj_error=reproj,
        n_inliers=len(detections) * 4,
        debug=debug,
        tag_pose_by_id=tag_pose_by_id,
    )


def native_tag_pose_fusion(detector: Any, tags: list[Any]) -> dict[str, Any]:
    detections = detections_from_tags(tags)
    candidates = []
    tag_pose_by_id: dict[int, dict[str, Any]] = {}
    for tag in tags:
        tag_id = int(tag.tag_id)
        pose = tag_pose_from_native_detection(tag)
        if pose is None:
            continue
        tag_rot, tag_tvec, tag_reproj = pose
        tag_pose_by_id[tag_id] = {
            "rot_mat": tag_rot,
            "tvec": tag_tvec,
            "reproj_error": tag_reproj,
        }
        cube_pose = cube_pose_from_tag_pose(detector, tag_id, tag_rot, tag_tvec)
        if cube_pose is None:
            continue
        cube_rot, cube_tvec = cube_pose
        candidates.append(
            {
                "tag_id": tag_id,
                "rot_mat": cube_rot,
                "tvec": cube_tvec,
                "reproj_error": tag_reproj,
            }
        )
    return result_from_cube_candidates(
        detector,
        detections,
        candidates,
        debug={"candidate_count": len(candidates)},
        tag_pose_by_id=tag_pose_by_id,
    )


def per_face_pnp_fusion(detector: Any, detections: list[tuple[int, np.ndarray]]) -> dict[str, Any]:
    face_to_detections: dict[str, list[tuple[int, np.ndarray]]] = {}
    for tag_id, corners in detections:
        for face_name, id_set in detector.face_id_sets.items():
            if int(tag_id) in id_set:
                face_to_detections.setdefault(face_name, []).append((tag_id, corners))
    candidates = []
    for face_name, face_detections in face_to_detections.items():
        pose = pnp_cube_pose_lm(detector, face_detections, use_ransac=False)
        if pose is None:
            continue
        rvec, tvec, reproj, _n = pose
        rot, _ = cv2.Rodrigues(rvec)
        candidates.append(
            {
                "face_name": face_name,
                "rot_mat": rot,
                "tvec": tvec,
                "reproj_error": reproj,
            }
        )
    return result_from_cube_candidates(
        detector,
        detections,
        candidates,
        debug={"face_candidate_count": len(candidates), "faces": sorted(face_to_detections.keys())},
    )


def create_temporal_estimator(detector: Any) -> TemporalTagPoseEstimator:
    return TemporalTagPoseEstimator(
        tag_size_m=float(detector.config.tag_size_mm) / 1000.0,
        pupil_to_object_corner_index=PUPIL_TO_CUBE_CORNER_INDEX,
        solvepnp_generic_flag=cv2.SOLVEPNP_IPPE,
        solvepnp_flag=cv2.SOLVEPNP_ITERATIVE,
        use_temporal_candidate_selection=True,
        use_solvepnp_refine_lm=True,
        translation_score_weight_deg_per_mm=1.0,
        reject_negative_camera_z=True,
    )


def temporal_tag_pose_fusion(
    detector: Any,
    tags: list[Any],
    temporal_estimator: TemporalTagPoseEstimator,
    *,
    camera_name: str,
) -> dict[str, Any]:
    detections = detections_from_tags(tags)
    candidates = []
    tag_pose_by_id: dict[int, dict[str, Any]] = {}
    for tag in tags:
        tag_id = int(tag.tag_id)
        solved = temporal_estimator.estimate_pose(
            camera_name=camera_name,
            tag_id=tag_id,
            corners_xy=np.asarray(tag.corners, dtype=np.float64).reshape(4, 2),
            k=np.asarray(detector.camera_matrix, dtype=np.float64),
            dist_coeffs=np.asarray(detector.dist_coeffs, dtype=np.float64),
        )
        if solved is None:
            continue
        tag_rot, tag_tvec_m, tag_reproj, _debug = solved
        tag_tvec_mm = np.asarray(tag_tvec_m, dtype=np.float64).reshape(3, 1) * 1000.0
        tag_pose_by_id[tag_id] = {
            "rot_mat": tag_rot,
            "tvec": tag_tvec_mm,
            "reproj_error": float(tag_reproj),
        }
        cube_pose = cube_pose_from_tag_pose(detector, tag_id, tag_rot, tag_tvec_mm)
        if cube_pose is None:
            continue
        cube_rot, cube_tvec = cube_pose
        candidates.append(
            {
                "tag_id": tag_id,
                "rot_mat": cube_rot,
                "tvec": cube_tvec,
                "reproj_error": float(tag_reproj),
            }
        )
    return result_from_cube_candidates(
        detector,
        detections,
        candidates,
        debug={"temporal_candidate_count": len(candidates)},
        tag_pose_by_id=tag_pose_by_id,
    )


def tag_pose_candidate_cube_consistency(detector: Any, tags: list[Any]) -> dict[str, Any]:
    detections = detections_from_tags(tags)
    obj_pts = tag_object_corners(float(detector.config.tag_size_mm) / 1000.0)
    k = np.asarray(detector.camera_matrix, dtype=np.float64)
    dist = np.asarray(detector.dist_coeffs, dtype=np.float64).reshape(-1, 1)
    candidates = []
    for tag in tags:
        tag_id = int(tag.tag_id)
        img_pts = reorder_pupil_corners_to_cube_order(np.asarray(tag.corners, dtype=np.float64))
        try:
            retval, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
                obj_pts,
                img_pts,
                k,
                dist,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            retval, rvecs, tvecs, reproj_errs = 0, [], [], None
        if int(retval) <= 0:
            continue
        reproj_arr = (
            np.asarray(reproj_errs, dtype=np.float64).reshape(-1)
            if reproj_errs is not None
            else np.full(len(rvecs), np.nan)
        )
        for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            tag_rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
            if not is_valid_rotation_matrix(tag_rot):
                continue
            tag_tvec_m = np.asarray(tvec, dtype=np.float64).reshape(3)
            if float(tag_tvec_m[2]) <= 0.0:
                continue
            tag_tvec_mm = tag_tvec_m.reshape(3, 1) * 1000.0
            cube_pose = cube_pose_from_tag_pose(detector, tag_id, tag_rot, tag_tvec_mm)
            if cube_pose is None:
                continue
            cube_rot, cube_tvec = cube_pose
            candidates.append(
                {
                    "tag_id": tag_id,
                    "candidate_index": int(idx),
                    "rot_mat": cube_rot,
                    "tvec": cube_tvec,
                    "reproj_error": float(reproj_arr[idx]) if idx < len(reproj_arr) and np.isfinite(reproj_arr[idx]) else None,
                }
            )
    if not candidates:
        return result_from_cube_candidates(detector, detections, [], debug={"raw_candidate_count": 0})

    best_support: list[dict[str, Any]] = []
    best_score = float("inf")
    for anchor in candidates:
        support = []
        score = 0.0
        for cand in candidates:
            rot_err = rotation_angle_deg(np.asarray(anchor["rot_mat"]), np.asarray(cand["rot_mat"]))
            trans_err = float(np.linalg.norm(np.asarray(anchor["tvec"]) - np.asarray(cand["tvec"])))
            if rot_err <= 35.0 and trans_err <= 25.0:
                support.append(cand)
                score += rot_err + trans_err
        score -= 1000.0 * len({int(c["tag_id"]) for c in support})
        if score < best_score:
            best_score = score
            best_support = support
    return result_from_cube_candidates(
        detector,
        detections,
        best_support,
        debug={"raw_candidate_count": len(candidates), "selected_candidate_count": len(best_support)},
    )


def rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    delta = np.asarray(rot_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    cos_angle = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def pose_xyz_rpy_from_result(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if not result.get("success", False):
        return None
    rvec = np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(result["tvec"], dtype=np.float64).reshape(3)
    rot, _ = cv2.Rodrigues(rvec)
    rpy = R.from_matrix(rot).as_euler("xyz", degrees=True)
    return tvec, np.asarray(rpy, dtype=np.float64)

