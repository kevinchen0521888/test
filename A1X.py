### ArUco Close-up Detection with Camera Calibration V3.0

import cv2
import numpy as np

class ArucoCloseUpDetector:
    def __init__(self, aruco_dict=cv2.aruco.DICT_4X4_50, marker_length=0.01):
        """
        :param aruco_dict: ArUco字典类型（推荐4x4小字典识别小标记）
        :param marker_length: ArUco标记边长（米），例如0.01表示1cm
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.parameters = cv2.aruco.DetectorParameters()
        self.marker_length = marker_length
        
        # 摄像头参数（从标定结果加载）
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 加载标定好的相机内参
        self._load_camera_calibration()
        
        # ========== 新增：记录上一次检测到的ID集合，用于控制控制台输出频率 ==========
        self._last_detected_ids = None
    
    def _load_camera_calibration(self):
        """加载相机标定参数"""
        # 相机内参矩阵
        self.camera_matrix = np.array ([
            [825.14514454, 0.0, 267.5745044],
            [0.0, 814.34909293, 249.58906419], 
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # 畸变系数 (k1, k2, p1, p2, k3)
        self.dist_coeffs = np.array(
            [-0.35715776,  0.01549191, -0.00584793,  0.01497875,  0.17125396]
        , dtype=np.float64)
        
        print("✅ 已加载相机内参:")
        print(f"   - 焦距: fx={self.camera_matrix[0,0]:.3f}, fy={self.camera_matrix[1,1]:.3f}")
        print(f"   - 光心: cx={self.camera_matrix[0,2]:.3f}, cy={self.camera_matrix[1,2]:.3f}")
        # 修复：将numpy数组元素转换为Python float
        print(f"   - 畸变系数: k1={float(self.dist_coeffs[0]):.4f}, k2={float(self.dist_coeffs[1]):.4f}")
        print(f"   - 重投影误差: 0.065519 像素")
    
    def set_camera_calibration(self, camera_matrix, dist_coeffs):
        """手动设置相机内参（可选）"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def calculate_sharpness(self, img):
        """计算图像清晰度（拉普拉斯方差）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def detect_aruco(self, img):
        """检测ArUco标记，返回角点、id、拒绝点"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        return corners, ids, rejected
    
    def draw_results(self, img, corners, ids, sharpness):
        """绘制检测结果和清晰度信息"""
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            # 估计位姿（使用已标定的相机内参）
            if self.camera_matrix is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length,
                    self.camera_matrix, self.dist_coeffs
                )
                for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    # 绘制坐标轴 - 修复OpenCV版本兼容性问题
                    try:
                        # 尝试新版本API
                        cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs, 
                                         rvec, tvec, 0.02)
                    except AttributeError:
                        # 回退到旧版本API
                        try:
                            cv2.aruco.drawAxis(img, self.camera_matrix, self.dist_coeffs, 
                                              rvec, tvec, 0.02)
                        except:
                            # 如果都失败，跳过绘制坐标轴
                            pass
                    
                    # 显示每个标记的距离信息
                    distance = np.linalg.norm(tvec)
                    # 修复：确保坐标是整数类型
                    corner_point = corners[i][0][0].astype(int)
                    cv2.putText(img, f"ID:{ids[i][0]} d:{distance*100:.1f}cm", 
                                tuple(corner_point), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # ========== 新增：在图像上显示每个标记的角点坐标和姿态数据 ==========
                    # 显示四个角点坐标（简化为左上角坐标示例，完整坐标可取消注释）
                    # 获取标记的第一个角点（作为参考位置）
                    first_corner = corners[i][0][0].astype(int)
                    # 在ID文本下方显示角点坐标（仅显示第一个角点，避免画面过挤）
                    cv2.putText(img, f"Corner1: ({first_corner[0]}, {first_corner[1]})",
                                (first_corner[0], first_corner[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # 显示旋转向量（rvec）和平移向量（tvec）的简化值
                    # 将numpy数组转换为可读字符串，保留2位小数
                    rvec_str = f"r: ({rvec[0][0]:.2f}, {rvec[0][1]:.2f}, {rvec[0][2]:.2f})"
                    tvec_str = f"t: ({tvec[0][0]:.2f}, {tvec[0][1]:.2f}, {tvec[0][2]:.2f})"
                    cv2.putText(img, rvec_str, (first_corner[0], first_corner[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.putText(img, tvec_str, (first_corner[0], first_corner[1] + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    # ========== 新增结束 ==========
                
                # ========== 新增：控制台输出完整信息（仅在ID集合发生变化时打印） ==========
                current_ids = set(ids.flatten()) if ids is not None else set()
                if self._last_detected_ids != current_ids:
                    self._last_detected_ids = current_ids.copy()
                    if ids is not None:
                        print("\n[ArUco 检测信息更新]")
                        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                            marker_id = ids[i][0]
                            # 获取四个角点坐标（原始浮点数）
                            corners_i = corners[i][0]
                            print(f"标记 ID: {marker_id}")
                            print(f"  四个角点坐标（像素，顺序为左上、右上、右下、左下）:")
                            for j, pt in enumerate(corners_i):
                                print(f"    点{j+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
                            print(f"  旋转向量 rvec (旋转轴*角度): ({rvec[0][0]:.4f}, {rvec[0][1]:.4f}, {rvec[0][2]:.4f})")
                            print(f"  平移向量 tvec (世界坐标, 米): ({tvec[0][0]:.4f}, {tvec[0][1]:.4f}, {tvec[0][2]:.4f})")
                            print(f"  距离: {np.linalg.norm(tvec):.4f} 米\n")
                # ========== 新增结束 ==========
        
        # 显示清晰度
        cv2.putText(img, f"Sharpness: {sharpness:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img
    
    def undistort_image(self, img):
        """对图像进行畸变校正"""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = img.shape[:2]
            # 获取最优化的相机内参和校正映射
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            # 校正图像
            dst = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
            # 裁剪图像（去除黑色边缘）
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            return dst
        return img
    
    def run_live(self, camera_id=0, undistort=False):
        """实时运行：手动调焦+ArUco检测
        
        :param camera_id: 摄像头ID
        :param undistort: 是否对输入图像进行畸变校正（可能影响实时性）
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        # 检查焦距控制支持
        focus_supported = False
        auto_focus = True
        focus_val = 0
        try:
            if cap.get(cv2.CAP_PROP_AUTOFOCUS) != -1:
                auto_focus = bool(cap.get(cv2.CAP_PROP_AUTOFOCUS))
                if cap.get(cv2.CAP_PROP_FOCUS) != -1:
                    focus_supported = True
                    focus_val = cap.get(cv2.CAP_PROP_FOCUS)
        except:
            pass
        
        print("\n=== 微距ArUco识别 ===")
        print("操作说明：")
        print("  '+' / '-' : 手动调节焦距（步长1）")
        print("  'f'       : 切换自动/手动对焦模式")
        print("  'a'       : 触发一次自动对焦")
        print("  'u'       : 切换畸变校正（实时性影响）")
        print("  'q'       : 退出")
        
        if not focus_supported:
            print("\n⚠️ 摄像头不支持焦距控制，请物理调整镜头距离或使用支持微距的设备。")
        
        print("\n📷 相机参数状态:")
        print(f"   - 已加载内参: {'是' if self.camera_matrix is not None else '否'}")
        print(f"   - 畸变校正: {'启用' if undistort else '禁用'}")
        
        # 显示OpenCV版本信息
        print(f"   - OpenCV版本: {cv2.__version__}")
        
        # 添加一个标志来控制循环
        running = True
        
        while running:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 可选：畸变校正
            if undistort and self.camera_matrix is not None:
                frame = self.undistort_image(frame)
            
            sharpness = self.calculate_sharpness(frame)
            corners, ids, _ = self.detect_aruco(frame)
            display = self.draw_results(frame.copy(), corners, ids, sharpness)
            
            # 显示焦距信息
            if focus_supported:
                mode = "Auto" if auto_focus else "Manual"
                cv2.putText(display, f"Focus: {mode} val={focus_val:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display, "Focus: Not supported", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 显示检测到的标记数量
            if ids is not None:
                cv2.putText(display, f"Detected: {len(ids)} markers", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示畸变校正状态
            if undistort:
                cv2.putText(display, "Undistort: ON", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Aruco Close-up", display)
            key = cv2.waitKey(1) & 0xFF
            
            # 处理按键事件
            if key == ord('q') or key == ord('Q'):  # 支持大小写q
                print("正在退出程序...")
                running = False  # 设置标志为False，退出循环
                break  # 立即跳出循环
            
            # 焦距控制
            if focus_supported:
                if key == ord('+') or key == ord('='):
                    if not auto_focus:
                        focus_val = min(focus_val + 1, 800)
                        cap.set(cv2.CAP_PROP_FOCUS, focus_val)
                        print(f"焦距: {focus_val:.1f}  清晰度: {sharpness:.1f}")
                elif key == ord('-') or key == ord('_'):
                    if not auto_focus:
                        focus_val = max(focus_val - 1, 0)
                        cap.set(cv2.CAP_PROP_FOCUS, focus_val)
                        print(f"焦距: {focus_val:.1f}  清晰度: {sharpness:.1f}")
                elif key == ord('f') or key == ord('F'):
                    auto_focus = not auto_focus
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if auto_focus else 0)
                    if auto_focus:
                        print("自动对焦模式")
                    else:
                        focus_val = cap.get(cv2.CAP_PROP_FOCUS)
                        print(f"手动对焦模式，当前焦距: {focus_val:.1f}")
                elif key == ord('a') or key == ord('A'):
                    # 触发自动对焦
                    if auto_focus:
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    else:
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                        cv2.waitKey(500)
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                        focus_val = cap.get(cv2.CAP_PROP_FOCUS)
                    print("自动对焦触发")
            elif key == ord('u') or key == ord('U'):
                undistort = not undistort
                print(f"畸变校正: {'启用' if undistort else '禁用'}")
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")
 
if __name__ == "__main__":
    detector = ArucoCloseUpDetector(aruco_dict=cv2.aruco.DICT_4X4_50, marker_length=0.01)  # 边长9mm
    detector.run_live(undistort=True)  # undistort=True可启用畸变校正