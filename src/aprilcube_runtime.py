from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def is_valid_rotation_matrix(rot: np.ndarray | None, det_tol: float = 0.2) -> bool:
    """Check whether rot is a valid right-handed rotation matrix."""
    if rot is None:
        return False

    rot = np.asarray(rot, dtype=np.float64)
    if rot.shape != (3, 3):
        return False
    if not np.all(np.isfinite(rot)):
        return False

    det = float(np.linalg.det(rot))
    if det <= 0.0 or abs(det - 1.0) > det_tol:
        return False

    ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3)))
    return ortho_err <= 0.2


def k_to_camera_params(k: np.ndarray) -> tuple[float, float, float, float]:
    k = np.asarray(k, dtype=np.float64)
    return (
        float(k[0, 0]),
        float(k[1, 1]),
        float(k[0, 2]),
        float(k[1, 2]),
    )


class AprilCubeTemporalPoseRuntime:
    """One cube-model runtime: native tag pose, optional temporal stabilization, cube fusion."""

    def __init__(
        self,
        *,
        detector: Any,
        native_detector: Any,
        pose_estimator: Any,
        use_clahe_for_native_detection: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self.detector = detector
        self.native_detector = native_detector
        self.pose_estimator = pose_estimator
        self.tag_size_m = float(detector.config.tag_size_mm) / 1000.0
        self.native_family = str(detector.config.dict_name)
        self.use_clahe_for_native_detection = bool(use_clahe_for_native_detection)
        self._clahe = None
        if self.use_clahe_for_native_detection:
            self._clahe = cv2.createCLAHE(
                clipLimit=float(clahe_clip_limit),
                tileGridSize=tuple(int(v) for v in clahe_tile_grid_size),
            )

    def _filter_valid_tags(self, tags: list[Any]) -> list[Any]:
        return [tag for tag in tags if int(tag.tag_id) in self.detector.valid_ids]

    def prepare_native_detection_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = np.asarray(gray, dtype=np.uint8)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        return gray

    def detect_native_apriltags_all(self, image: np.ndarray) -> list[Any]:
        gray = self.prepare_native_detection_gray(image)
        camera_params = k_to_camera_params(np.asarray(self.detector.camera_matrix, dtype=np.float64))
        tags = self.native_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=self.tag_size_m,
        )
        return list(tags)

    def detect_native_apriltags(self, image: np.ndarray) -> list[Any]:
        return self._filter_valid_tags(self.detect_native_apriltags_all(image))

    def tag_pose_from_native_detection(
        self,
        tag: Any,
    ) -> tuple[np.ndarray, np.ndarray, float | None] | None:
        pose_R = getattr(tag, "pose_R", None)
        pose_t = getattr(tag, "pose_t", None)
        if pose_R is None or pose_t is None:
            return None

        pose_R = np.asarray(pose_R, dtype=np.float64).reshape(3, 3)
        if not is_valid_rotation_matrix(pose_R):
            return None

        pose_t_mm = np.asarray(pose_t, dtype=np.float64).reshape(3, 1) * 1000.0
        pose_err = getattr(tag, "pose_err", None)
        reproj_error = float(pose_err) if pose_err is not None else None
        return pose_R, pose_t_mm, reproj_error

    def tag_pose_from_temporal_estimator(
        self,
        camera_name: str,
        tag: Any,
    ) -> tuple[np.ndarray, np.ndarray, float | None] | None:
        solved = self.pose_estimator.estimate_pose(
            camera_name=camera_name,
            tag_id=int(tag.tag_id),
            corners_xy=np.asarray(tag.corners, dtype=np.float64).reshape(4, 2),
            k=np.asarray(self.detector.camera_matrix, dtype=np.float64),
            dist_coeffs=np.asarray(self.detector.dist_coeffs, dtype=np.float64)
            if self.detector.dist_coeffs is not None
            else None,
        )
        if solved is None:
            return None

        pose_R, pose_t, reproj_error_px, _debug_info = solved
        pose_t_mm = np.asarray(pose_t, dtype=np.float64).reshape(3, 1) * 1000.0
        return pose_R, pose_t_mm, float(reproj_error_px)

    def build_tag_to_cube_transform(
        self,
        tag_id: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        corners_3d = self.detector.tag_corner_map.get(int(tag_id))
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

        center_cube = np.mean(np.asarray(corners_3d, dtype=np.float64), axis=0).reshape(3, 1)
        return rot_cube_tag, center_cube

    def cube_pose_from_tag_pose(
        self,
        tag_id: int,
        tag_rot_mat: np.ndarray,
        tag_tvec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        tag_to_cube = self.build_tag_to_cube_transform(tag_id)
        if tag_to_cube is None:
            return None

        rot_cube_tag, center_cube = tag_to_cube
        rot_tag_cube = rot_cube_tag.T
        center_tag = -rot_tag_cube @ center_cube

        rot_cam_tag = np.asarray(tag_rot_mat, dtype=np.float64).reshape(3, 3)
        tag_tvec = np.asarray(tag_tvec, dtype=np.float64).reshape(3, 1)

        rot_cam_cube = rot_cam_tag @ rot_tag_cube
        center_cam = rot_cam_tag @ center_tag + tag_tvec

        if not is_valid_rotation_matrix(rot_cam_cube):
            return None
        return rot_cam_cube, center_cam

    @staticmethod
    def tag_z_points_to_cube_interior(
        tag_rot_mat: np.ndarray,
        tag_tvec: np.ndarray,
        cube_center_cam: np.ndarray,
    ) -> bool:
        tag_rot_mat = np.asarray(tag_rot_mat, dtype=np.float64).reshape(3, 3)
        tag_tvec = np.asarray(tag_tvec, dtype=np.float64).reshape(3)
        cube_center_cam = np.asarray(cube_center_cam, dtype=np.float64).reshape(3)

        z_axis_cam = tag_rot_mat[:, 2]
        to_cube_center = cube_center_cam - tag_tvec
        return float(np.dot(z_axis_cam, to_cube_center)) > 0.0

    @staticmethod
    def average_rotations(rot_mats: list[np.ndarray], weights: np.ndarray) -> np.ndarray | None:
        if not rot_mats:
            return None

        accum = np.zeros((3, 3), dtype=np.float64)
        for rot, weight in zip(rot_mats, weights):
            accum += float(weight) * np.asarray(rot, dtype=np.float64)

        u, _s, vt = np.linalg.svd(accum)
        rot_avg = u @ vt
        if np.linalg.det(rot_avg) < 0.0:
            u[:, -1] *= -1.0
            rot_avg = u @ vt

        if not is_valid_rotation_matrix(rot_avg):
            return None
        return rot_avg

    def fuse_cube_pose_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray] | None:
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
        weights_arr /= np.sum(weights_arr)

        rot_avg = self.average_rotations(rot_mats, weights_arr)
        if rot_avg is None:
            return None

        t_avg = np.zeros((3, 1), dtype=np.float64)
        for weight, tvec in zip(weights_arr, tvecs):
            t_avg += float(weight) * tvec
        return rot_avg, t_avg

    def compute_reprojection_error(
        self,
        detections: list[tuple[int, np.ndarray]],
        cube_rvec: np.ndarray,
        cube_tvec: np.ndarray,
    ) -> float:
        if not detections:
            return float("inf")

        object_points = np.vstack([
            np.asarray(self.detector.tag_corner_map[int(tag_id)], dtype=np.float64)
            for tag_id, _corners in detections
        ])
        image_points = np.vstack([
            np.asarray(corners, dtype=np.float64)
            for _tag_id, corners in detections
        ])

        projected, _ = cv2.projectPoints(
            object_points,
            cube_rvec,
            cube_tvec,
            self.detector.camera_matrix,
            self.detector.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(image_points - projected, axis=1)))

    def process_frame(
        self,
        camera_name: str,
        image: np.ndarray,
        native_tags: list[Any] | None = None,
    ) -> dict[str, Any]:
        result = {
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
            "tag_z_inward_count": 0,
            "tag_z_invalid_count": 0,
            "tag_pose_by_id": {},
        }

        if native_tags is None:
            native_tags = self.detect_native_apriltags(image)
        else:
            native_tags = self._filter_valid_tags(native_tags)

        detections = [
            (int(tag.tag_id), np.asarray(tag.corners, dtype=np.float64).reshape(4, 2))
            for tag in native_tags
        ]
        result["detections"] = detections
        result["n_tags"] = len(detections)
        result["tag_ids"] = [tag_id for tag_id, _ in detections]

        for tag_id, _corners in detections:
            for face_name, id_set in self.detector.face_id_sets.items():
                if tag_id in id_set:
                    result["visible_faces"].add(face_name)

        if not detections:
            return self.detector._store_latest(result, image)

        use_temporal_pose = len(result["visible_faces"]) == 1
        first_pass_candidates: list[dict[str, Any]] = []
        tag_meas: list[dict[str, Any]] = []
        for tag in native_tags:
            tag_id = int(tag.tag_id)
            if use_temporal_pose:
                pose = self.tag_pose_from_temporal_estimator(camera_name, tag)
            else:
                pose = self.tag_pose_from_native_detection(tag)
            if pose is None:
                continue

            rot_mat, tvec, reproj = pose
            corners_2d = np.asarray(tag.corners, dtype=np.float64).reshape(4, 2)
            tag_meas.append(
                {
                    "tag_id": tag_id,
                    "corners_2d": corners_2d,
                    "rot_mat": rot_mat,
                    "tvec": tvec,
                    "reproj_error": reproj,
                }
            )
            cube_pose = self.cube_pose_from_tag_pose(tag_id, rot_mat, tvec)
            if cube_pose is None:
                continue
            cube_rot_mat, cube_tvec = cube_pose
            first_pass_candidates.append(
                {
                    "tag_id": tag_id,
                    "rot_mat": cube_rot_mat,
                    "tvec": cube_tvec,
                    "reproj_error": reproj,
                }
            )

        preliminary_cube = self.fuse_cube_pose_candidates(first_pass_candidates)
        preliminary_center = None
        if preliminary_cube is not None:
            preliminary_center = np.asarray(preliminary_cube[1], dtype=np.float64).reshape(3)

        chosen_candidates: list[dict[str, Any]] = []
        inward_count = 0
        invalid_count = 0
        for meas in tag_meas:
            tag_id = meas["tag_id"]
            inward_ok = None
            if preliminary_center is not None:
                inward_ok = self.tag_z_points_to_cube_interior(
                    meas["rot_mat"],
                    meas["tvec"],
                    preliminary_center,
                )
                if inward_ok:
                    inward_count += 1
                else:
                    invalid_count += 1

            result["tag_pose_by_id"][tag_id] = {
                "rot_mat": meas["rot_mat"],
                "tvec": meas["tvec"],
                "reproj_error": meas["reproj_error"],
                "z_inward": inward_ok,
            }

            if inward_ok is False:
                continue

            cube_pose = self.cube_pose_from_tag_pose(tag_id, meas["rot_mat"], meas["tvec"])
            if cube_pose is None:
                continue
            cube_rot_mat, cube_tvec = cube_pose
            chosen_candidates.append(
                {
                    "tag_id": tag_id,
                    "rot_mat": cube_rot_mat,
                    "tvec": cube_tvec,
                    "reproj_error": meas["reproj_error"],
                }
            )

        result["tag_z_inward_count"] = inward_count
        result["tag_z_invalid_count"] = invalid_count

        if not chosen_candidates:
            chosen_candidates = first_pass_candidates
        final_cube = self.fuse_cube_pose_candidates(chosen_candidates)
        if final_cube is None:
            return self.detector._store_latest(result, image)

        cube_rot_mat, cube_tvec = final_cube
        cube_rvec, _ = cv2.Rodrigues(cube_rot_mat)
        result["success"] = True
        result["rvec"] = cube_rvec
        result["tvec"] = cube_tvec
        result["n_inliers"] = len(chosen_candidates) * 4
        result["reproj_error"] = self.compute_reprojection_error(
            detections=detections,
            cube_rvec=cube_rvec,
            cube_tvec=cube_tvec,
        )

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = cube_rot_mat
        T[:3, 3] = cube_tvec.reshape(3)
        result["T"] = T

        self.detector.prev_rvec = cube_rvec.copy()
        self.detector.prev_tvec = cube_tvec.copy()
        return self.detector._store_latest(result, image)

    def draw_tag_frame_projection(
        self,
        img: np.ndarray,
        pose_R: np.ndarray,
        pose_t: np.ndarray,
        axis_length_mm: float,
    ) -> None:
        pose_R = np.asarray(pose_R, dtype=np.float64)
        pose_t = np.asarray(pose_t, dtype=np.float64).reshape(3, 1)

        if not is_valid_rotation_matrix(pose_R):
            return

        obj_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_length_mm, 0.0, 0.0],
                [0.0, axis_length_mm, 0.0],
                [0.0, 0.0, axis_length_mm],
            ],
            dtype=np.float64,
        )

        rvec, _ = cv2.Rodrigues(pose_R)
        dist_coeffs = self.detector.dist_coeffs
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5, dtype=np.float64)

        img_pts, _ = cv2.projectPoints(
            objectPoints=obj_pts,
            rvec=rvec,
            tvec=pose_t,
            cameraMatrix=np.asarray(self.detector.camera_matrix, dtype=np.float64),
            distCoeffs=np.asarray(dist_coeffs, dtype=np.float64),
        )
        img_pts = np.round(img_pts.reshape(-1, 2)).astype(np.int32)

        origin = tuple(img_pts[0])
        pt_x = tuple(img_pts[1])
        pt_y = tuple(img_pts[2])
        pt_z = tuple(img_pts[3])

        cv2.arrowedLine(img, origin, pt_x, (0, 0, 255), 4, tipLength=0.25)
        cv2.arrowedLine(img, origin, pt_y, (0, 255, 0), 4, tipLength=0.25)
        cv2.arrowedLine(img, origin, pt_z, (255, 0, 0), 4, tipLength=0.25)

        cv2.putText(img, "x", pt_x, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(img, "y", pt_y, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(img, "z", pt_z, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

    def draw_detected_tag_visuals(
        self,
        img: np.ndarray,
        result: dict[str, Any] | None,
        *,
        draw_tag_frame_2d: bool = True,
        tag_axis_length_scale: float = 0.8,
    ) -> np.ndarray:
        if not result:
            return img

        detections = result.get("detections", [])
        if not detections:
            return img

        out = img.copy()
        tag_pose_by_id = result.get("tag_pose_by_id", {})
        tag_axis_length_mm = float(self.detector.config.tag_size_mm) * float(tag_axis_length_scale)

        for tag_id, corners_2d in detections:
            corners = np.round(np.asarray(corners_2d, dtype=np.float64)).astype(np.int32)
            if corners.shape != (4, 2):
                continue

            center_xy = np.round(np.mean(corners, axis=0)).astype(np.int32)
            c_x, c_y = int(center_xy[0]), int(center_xy[1])

            cv2.putText(
                out,
                f"ID:{int(tag_id)}",
                (c_x - 18, c_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.circle(out, (c_x, c_y), 4, (0, 0, 255), -1)

            if not draw_tag_frame_2d:
                cv2.polylines(out, [corners], True, (0, 255, 0), 4)
                continue

            tag_pose = tag_pose_by_id.get(int(tag_id))
            if tag_pose is None:
                cv2.polylines(out, [corners], True, (0, 0, 255), 4)
                continue

            tag_rot_mat = np.asarray(tag_pose["rot_mat"], dtype=np.float64)
            tag_tvec = np.asarray(tag_pose["tvec"], dtype=np.float64)
            tag_is_inward = tag_pose.get("z_inward", None)

            border_color = (0, 255, 0) if tag_is_inward is not False else (0, 0, 255)
            cv2.polylines(out, [corners], True, border_color, 4)

            self.draw_tag_frame_projection(
                img=out,
                pose_R=tag_rot_mat,
                pose_t=tag_tvec,
                axis_length_mm=tag_axis_length_mm,
            )

            if tag_is_inward is False:
                cv2.putText(
                    out,
                    "z-out",
                    (c_x - 20, c_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                )

        return out
