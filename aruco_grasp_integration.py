"""
ArUco 姿态计算与机械臂抓取插销拔除集成模块
用于集成到 main_v3.py 主程序中

使用方法:
    1. 在主类中导入: from ArUco.aruco_grasp_integration import ArucoGraspController
    2. 初始化: self.aruco_grasp = ArucoGraspController()
    3. 调用: self.aruco_grasp.run_grasp_and_pull_process()
"""

import cv2
import numpy as np
import time
import json
from typing import Tuple, Optional, List
from coordinate_transform import transform_vision_to_robot


class ArucoGraspController:
    """ArUco 视觉引导抓取控制器"""
    
    def __init__(self, config=None):
        """
        初始化 ArUco 抓取控制器
        
        参数:
            config: 配置字典，可选
        """
        # 默认配置
        self.config = config or self._get_default_config()
        
        # 初始化 ArUco 检测器
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.config['aruco_dict_name'])
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # 相机参数
        self.camera_matrix = np.array(self.config['camera_matrix'], dtype=np.float64)
        self.dist_coeffs = np.array(self.config['dist_coeffs'], dtype=np.float64)
        
        # 抓取参数
        self.approach_height = self.config['approach_height']
        self.pull_distance = self.config['pull_distance']
        self.grasp_force = self.config['grasp_force']
        
        # 状态
        self.robot_controller = None
        self.last_robot_point = None
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'aruco_dict_name': 'DICT_4X4_50',
            'marker_length': 0.009,  # 9mm
            'target_marker_id': 0,
            'camera_matrix': [
                [663.8079926, 0., 319.2422067],
                [0., 656.01709879, 239.22781472],
                [0., 0., 1.]
            ],
            'dist_coeffs': [
                -2.48957892e-02, -1.49252987e+00, 1.96323976e-03,
                2.66920504e-02, 3.50378383e+00
            ],
            'approach_height': 0.05,  # 50mm
            'pull_distance': 0.10,     # 100mm
            'grasp_force': 50,
            'correspondence_file': 'p0.json'
        }
    
    def detect_aruco_pose(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测 ArUco 标记并估计姿态
        
        参数:
            image: 输入图像 (BGR)
        
        返回:
            (rvec, tvec, corners) 旋转向量、平移向量、角点
            如果未检测到，返回 (None, None, None)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测标记
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.aruco_dict, 
            parameters=self.detector_params
        )
        
        if ids is None:
            return None, None, None
        
        # 查找目标标记
        target_ids = np.where(ids.flatten() == self.config['target_marker_id'])[0]
        if len(target_ids) == 0:
            return None, None, None
        
        target_idx = target_ids[0]
        target_corners = [corners[target_idx]]
        
        # 姿态估计
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            target_corners,
            self.config['marker_length'],
            self.camera_matrix,
            self.dist_coeffs
        )
        
        return rvec[0], tvec[0], corners[target_idx][0]
    
    def calculate_grasp_pose(self, image: np.ndarray, corners: np.ndarray) -> List[float]:
        """
        计算抓取位姿
        
        参数:
            image: 当前图像
            corners: ArUco 标记角点 (4x2)
        
        返回:
            robot_point: [x, y, z, rx, ry, rz] 机器人坐标
        """
        # 计算标记中心像素坐标
        center_x, center_y = corners.mean(axis=0)
        
        # 构建视觉坐标点
        vision_point = [
            float(center_x), 
            float(center_y), 
            80.0,  # Z 坐标 (mm)
            180.0, 0.0, -180.0  # 姿态角 (度)
        ]
        
        # 坐标转换
        try:
            robot_point = transform_vision_to_robot(
                vision_point, 
                self.config['correspondence_file']
            )
            self.last_robot_point = robot_point
            return robot_point
        except Exception as e:
            print(f"坐标转换失败：{e}")
            return None
    
    def move_to_grasp_pose(self, robot_controller, grasp_pose: List[float], 
                           queue_tag: int = 1) -> bool:
        """
        移动机械臂到抓取位姿
        
        参数:
            robot_controller: 机械臂控制器实例
            grasp_pose: 抓取位姿 [x, y, z, rx, ry, rz]
            queue_tag: 队列标签
        
        返回:
            是否成功
        """
        if grasp_pose is None:
            return False
        
        # 计算接近点（目标上方）
        approach_pose = grasp_pose.copy()
        approach_pose[2] += self.approach_height * 1000  # 转换为 mm
        
        try:
            # 1. 移动到接近点
            robot_controller.RobotMove([approach_pose], queue_tag)
            time.sleep(2)
            
            # 2. 移动到抓取点
            robot_controller.RobotMove([grasp_pose], queue_tag + 1)
            time.sleep(1)
            
            # 3. 闭合夹爪
            robot_controller.RobotGrab(0)  # 通道 0
            time.sleep(0.5)
            
            # 4. 提升
            robot_controller.RobotMove([approach_pose], queue_tag + 2)
            time.sleep(1)
            
            return True
        except Exception as e:
            print(f"移动失败：{e}")
            return False
    
    def pull_pin(self, robot_controller, grasp_pose: List[float], 
                 pull_distance: float = None, queue_tag: int = 5) -> bool:
        """
        执行插销拔除动作
        
        参数:
            robot_controller: 机械臂控制器
            grasp_pose: 抓取位姿
            pull_distance: 拔除距离 (米)
            queue_tag: 队列标签
        
        返回:
            是否成功
        """
        if pull_distance is None:
            pull_distance = self.pull_distance
        
        if grasp_pose is None:
            return False
        
        try:
            # 1. 计算拔除后的位置
            pull_pose = grasp_pose.copy()
            pull_pose[2] += pull_distance * 1000  # 转换为 mm
            
            # 2. 执行拔除
            robot_controller.RobotMove([pull_pose], queue_tag)
            time.sleep(3)
            
            # 3. 移动到放置点 (示例位置，需根据实际情况修改)
            place_pose = [-480.0, 813.99, -28.28, -124.0, 89.0, 56.0]
            robot_controller.RobotMove([place_pose], queue_tag + 1)
            time.sleep(2)
            
            # 4. 松开夹爪
            robot_controller.RobotUnGrab(0)
            time.sleep(0.5)
            
            return True
        except Exception as e:
            print(f"拔除失败：{e}")
            return False
    
    def run_grasp_and_pull_process(self, robot_controller, image: np.ndarray, 
                                   logger=None) -> bool:
        """
        执行完整的抓取和拔除流程
        
        参数:
            robot_controller: 机械臂控制器实例
            image: 当前相机图像
            logger: 日志记录器
        
        返回:
            是否成功完成
        """
        log = logger.info if logger else print
        
        log("🎯 开始 ArUco 视觉引导抓取流程...")
        
        # 步骤 1: ArUco 检测
        log("步骤 1: 检测 ArUco 标记")
        rvec, tvec, corners = self.detect_aruco_pose(image)
        
        if tvec is None:
            log("❌ 未检测到 ArUco 标记")
            return False
        
        log(f"✅ 检测到 ArUco 标记，距离：{np.linalg.norm(tvec):.3f} m")
        
        # 步骤 2: 计算抓取位姿
        log("步骤 2: 计算抓取位姿")
        grasp_pose = self.calculate_grasp_pose(image, corners)
        
        if grasp_pose is None:
            log("❌ 坐标转换失败")
            return False
        
        log(f"✅ 抓取位姿：{grasp_pose}")
        
        # 步骤 3: 移动到抓取点并抓取
        log("步骤 3: 执行抓取")
        if not self.move_to_grasp_pose(robot_controller, grasp_pose, queue_tag=1):
            log("❌ 抓取失败")
            return False
        
        log("✅ 抓取完成")
        
        # 步骤 4: 拔除插销
        log("步骤 4: 执行拔除")
        if not self.pull_pin(robot_controller, grasp_pose):
            log("❌ 拔除失败")
            return False
        
        log("✅ 插销拔除完成!")
        return True
    
    def draw_detection_result(self, image: np.ndarray, corners: np.ndarray, 
                               tvec: np.ndarray) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        参数:
            image: 输入图像
            corners: ArUco 角点
            tvec: 平移向量
        
        返回:
            标注后的图像
        """
        display = image.copy()
        
        if corners is not None:
            # 绘制角点
            corners_int = corners.astype(int)
            cv2.polylines(display, [corners_int], True, (0, 255, 0), 2)
            
            # 绘制中心点
            center = corners.mean(axis=0).astype(int)
            cv2.circle(display, tuple(center), 5, (0, 0, 255), -1)
            
            # 显示距离
            if tvec is not None:
                distance = np.linalg.norm(tvec)
                cv2.putText(display, f"Dist: {distance*100:.1f} cm", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return display


# ==================== 集成到 main_v3.py 的示例 ====================

def integrate_to_main_example():
    """
    这是一个示例，展示如何将 ArucoGraspController 集成到 main_v3.py 中
    
    在 main_v3.py 的 ImageProcessorApp 类中添加:
    """
    
    # 1. 在 __init__ 方法中添加:
    # from ArUco.aruco_grasp_integration import ArucoGraspController
    # self.aruco_grasp = ArucoGraspController()
    
    # 2. 添加新的步骤函数:
    """
    def step6_aruco_grasp(self):
        '''步骤 6: ArUco 姿态计算与抓取插销拔除'''
        self.logger.info("开始 ArUco 姿态计算与抓取")
        
        # 采集图像
        ret, frame = self.capture[self.list_num].read()
        if not ret:
            self.logger.error("图像采集失败")
            return
        
        # 执行抓取流程
        if self.m_robot:
            success = self.aruco_grasp.run_grasp_and_pull_process(
                self.m_robot, 
                frame, 
                self.logger
            )
            
            if success:
                self.logger.info("✅ 抓取拔除完成")
            else:
                self.logger.error("❌ 抓取拔除失败")
        else:
            self.logger.error("机械臂未连接")
    """
    
    # 3. 在 UI 中添加按钮绑定:
    # self.step6_aruco_pushButton.clicked.connect(self.step6_aruco_grasp)
    
    pass


if __name__ == "__main__":
    # 测试代码
    print("ArUco 抓取控制器测试")
    
    # 创建控制器
    controller = ArucoGraspController()
    
    # 测试图像
    test_image = cv2.imread('test_aruco.jpg')
    if test_image is None:
        print("请提供测试图像 test_aruco.jpg")
    else:
        # 检测
        rvec, tvec, corners = controller.detect_aruco_pose(test_image)
        
        if tvec is not None:
            print(f"检测到 ArUco 标记")
            print(f"  距离：{np.linalg.norm(tvec):.3f} m")
            print(f"  位置：x={tvec[0]:.3f}, y={tvec[1]:.3f}, z={tvec[2]:.3f}")
            
            # 计算抓取位姿
            grasp_pose = controller.calculate_grasp_pose(test_image, corners)
            if grasp_pose:
                print(f"  抓取位姿：{grasp_pose}")
        
        # 显示结果
        display = controller.draw_detection_result(test_image, corners, tvec)
        cv2.imshow("Detection Result", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
