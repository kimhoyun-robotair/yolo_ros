# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import numpy as np
from typing import List, Tuple

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

# lifecycle 도입
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

# 토픽간 시간 오차 방지를 위해 메시지 필터 도입
import message_filters
from cv_bridge import CvBridge
# 좌표 변환을 위해 tf2 라이브러리 사용
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import KeyPoint3DArray
from yolo_msgs.msg import BoundingBox3D


class Detect3DNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("bbox3d_node")

        # parameters
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter(
            "depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT
        )
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # aux
        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.maximum_detection_threshold = (
            self.get_parameter("maximum_detection_threshold")
            .get_parameter_value()
            .double_value
        )
        self.depth_image_units_divisor = (
            self.get_parameter("depth_image_units_divisor")
            .get_parameter_value()
            .integer_value
        )
        dimg_reliability = (
            self.get_parameter("depth_image_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        dinfo_reliability = (
            self.get_parameter("depth_info_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self._pub = self.create_publisher(DetectionArray, "detections_3d", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # subs
        # 시간 오차 최소화를 위해서 메시지 필터를 이용해서 subscirber들을 모두 통합
        self.depth_sub = message_filters.Subscriber(
            self, Image, "depth_image", qos_profile=self.depth_image_qos_profile
        )
        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "depth_info", qos_profile=self.depth_info_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections"
        )
        
        # 싱크로나이즈 하고, 콜백 부여
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.on_detections)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.depth_sub.sub)
        self.destroy_subscription(self.depth_info_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        del self.tf_listener

        self.destroy_publisher(self._pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def on_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> None:
        # 최종 yolo 탐지 결과를 publish 하는 콜백
        new_detections_msg = DetectionArray() # 탐지 결과 배열 (여러 객체 탐지할 경우)
        new_detections_msg.header = detections_msg.header
        new_detections_msg.detections = self.process_detections(
            depth_msg, depth_info_msg, detections_msg
        )
        self._pub.publish(new_detections_msg)

    # yolo 탐지를 수행하는 노드
    # yolo 탐지 및 3차원 bbox 확장을 수행해서 최종 결과를 return 하는 함수
    def process_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> List[Detection]:

        # check if there are detections
        if not detections_msg.detections:
            return []

        transform = self.get_transform(depth_info_msg.header.frame_id) # TF 변환을 통해 평행/회전 변환 정보 저장

        if transform is None:
            return []

        new_detections = []
        depth_image = self.cv_bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        ) # depth image를 ROS2에서 cv2로 변환

        for detection in detections_msg.detections:
            # depth image와 yolo 탐지 결과르 통해 bbox를 3차원으로 확장
            bbox3d, avg_depth = self.convert_bb_to_3d(depth_image, depth_info_msg, detection)

            if bbox3d is not None:
                new_detections.append(detection)

                bbox3d = Detect3DNode.transform_3d_box(bbox3d, transform[0], transform[1])
                bbox3d.frame_id = self.target_frame
                detection.depth = avg_depth
                new_detections[-1].bbox3d = bbox3d

                if detection.keypoints.data:
                    keypoints3d = self.convert_keypoints_to_3d(
                        depth_image, depth_info_msg, detection
                    )
                    keypoints3d = Detect3DNode.transform_3d_keypoints(
                        keypoints3d, transform[0], transform[1]
                    )
                    keypoints3d.frame_id = self.target_frame
                    new_detections[-1].keypoints3d = keypoints3d

        return new_detections

    def convert_bb_to_3d( # 2차원 bbox를 3차원 bbox 확장
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> BoundingBox3D:

        center_x = int(detection.bbox.center.position.x)
        center_y = int(detection.bbox.center.position.y)
        size_x = int(detection.bbox.size.x)
        size_y = int(detection.bbox.size.y)

        if detection.mask.data:
            # crop depth image by mask
            mask_array = np.array(
                [[int(ele.x), int(ele.y)] for ele in detection.mask.data]
            )
            mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(mask_array, dtype=np.int32)], 255)
            roi = cv2.bitwise_and(depth_image, depth_image, mask=mask)

        else:
            # crop depth image by the 2d BB
            u_min = max(center_x - size_x // 2, 0)
            u_max = min(center_x + size_x // 2, depth_image.shape[1] - 1)
            v_min = max(center_y - size_y // 2, 0)
            v_max = min(center_y + size_y // 2, depth_image.shape[0] - 1)

            roi = depth_image[v_min:v_max, u_min:u_max]

            # 물체까지의 depth 추정을 위해서 별도의 ROI 설정
            width_crop = int(size_x*0.3)
            height_crop = int(size_y*0.3)

            u_min_for_depth = max(center_x - size_x // 2 + width_crop, 0)
            u_max_for_depth = min(center_x + size_x // 2 - width_crop, depth_image.shape[1] - 1)
            v_min_for_depth = max(center_y - size_y // 2 + height_crop, 0)
            v_max_for_depth = min(center_y + size_y // 2 - height_crop, depth_image.shape[0] - 1)

            roi_for_depth = depth_image[v_min_for_depth:v_max_for_depth, u_min_for_depth:u_max_for_depth]

        roi = roi / self.depth_image_units_divisor  # convert to meters
        roi_for_depth = roi_for_depth / self.depth_image_units_divisor # conver to meters for depth estimation
        # 이중에서 유효한 depth값만 추정
        valid_depth = roi_for_depth[np.isfinite(roi_for_depth) & (roi_for_depth>0)]
        # outlier 제거
        z_median = np.median(valid_depth)
        z_std = np.std(valid_depth)
        z_filterd = float(np.median(valid_depth[(valid_depth > z_median - 1.5*z_std) & (valid_depth < z_median + 1.5*z_std)]))

        if not np.any(roi):
            return None, None

        # find the z coordinate on the 3D BB
        if detection.mask.data:
            roi = roi[roi > 0]
            bb_center_z_coord = np.median(roi)

        else:
            bb_center_z_coord = (
                depth_image[int(center_y)][int(center_x)] / self.depth_image_units_divisor
            )

        z_diff = np.abs(roi - bb_center_z_coord)
        mask_z = z_diff <= self.maximum_detection_threshold
        if not np.any(mask_z):
            return None, None
        roi = roi[mask_z]
        z_min, z_max = np.min(roi), np.max(roi)
        z = (z_max + z_min) / 2

        if z == 0:
            return None, None

        # project from image to world space
        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]
        x = z * (center_x - px) / fx
        y = z * (center_y - py) / fy
        w = z * (size_x / fx)
        h = z * (size_y / fy)

        # create 3D BB
        msg = BoundingBox3D()
        msg.center.position.x = x
        msg.center.position.y = y
        msg.center.position.z = z
        msg.size.x = w
        msg.size.y = h
        msg.size.z = float(z_max - z_min)

        return msg, z_filterd

    def convert_keypoints_to_3d( # yolo의 2D keypoint (예: human pose)를 3차원으로 변환
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> KeyPoint3DArray:

        # build an array of 2d keypoints
        keypoints_2d = np.array(
            [[p.point.x, p.point.y] for p in detection.keypoints.data], dtype=np.int16
        )
        u = np.array(keypoints_2d[:, 1]).clip(0, depth_info.height - 1)
        v = np.array(keypoints_2d[:, 0]).clip(0, depth_info.width - 1)

        # sample depth image and project to 3D
        z = depth_image[u, v]
        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]
        x = z * (v - px) / fx
        y = z * (u - py) / fy
        points_3d = (
            np.dstack([x, y, z]).reshape(-1, 3) / self.depth_image_units_divisor
        )  # convert to meters

        # generate message
        msg_array = KeyPoint3DArray()
        for p, d in zip(points_3d, detection.keypoints.data):
            if not np.isnan(p).any():
                msg = KeyPoint3D()
                msg.point.x = p[0]
                msg.point.y = p[1]
                msg.point.z = p[2]
                msg.id = d.id
                msg.score = d.score
                msg_array.data.append(msg)

        return msg_array

    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]: # TF 변환 조회
        # transform position from image frame to target_frame
        rotation = None
        translation = None

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rclpy.time.Time()
            )

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            rotation = np.array(
                [
                    transform.transform.rotation.w,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                ]
            )

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None

    @staticmethod
    def transform_3d_box( # 쿼터니언을 통해 3차원 bbox의 중심을 TF 좌표계로 변환
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> BoundingBox3D:

        # position
        position = (
            Detect3DNode.qv_mult(
                rotation,
                np.array(
                    [
                        bbox.center.position.x,
                        bbox.center.position.y,
                        bbox.center.position.z,
                    ]
                ),
            )
            + translation
        )

        bbox.center.position.x = position[0]
        bbox.center.position.y = position[1]
        bbox.center.position.z = position[2]

        # size
        size = Detect3DNode.qv_mult(
            rotation, np.array([bbox.size.x, bbox.size.y, bbox.size.z])
        )

        bbox.size.x = abs(size[0])
        bbox.size.y = abs(size[1])
        bbox.size.z = abs(size[2])

        return bbox

    @staticmethod
    def transform_3d_keypoints( # 3D Keypoint들도 TF 좌표계로 중심점을 변환
        keypoints: KeyPoint3DArray,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> KeyPoint3DArray:

        for point in keypoints.data:
            position = (
                Detect3DNode.qv_mult(
                    rotation, np.array([point.point.x, point.point.y, point.point.z])
                )
                + translation
            )

            point.point.x = position[0]
            point.point.y = position[1]
            point.point.z = position[2]

        return keypoints

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray: # 쿼터니언과 벡터간 곱셉을 수행하는 함수 (cross product)
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)


def main():
    rclpy.init()
    node = Detect3DNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
