import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import platform
import psutil
import GPUtil
import cv2

from PIL import Image

from PyQt5.QtCore import Qt, QTimer, QCoreApplication, QDir
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QCheckBox, QLineEdit, QLabel, QRadioButton, QHeaderView, QGroupBox, QGridLayout, QTextEdit, QComboBox, 
    QMessageBox, QTableWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QTreeWidget, QListWidget, QStackedLayout, QMenu, QFrame, QSpacerItem, QSizePolicy,
    QFileDialog)
from PyQt5.QtGui import QIcon

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
from model_patch_pixel import CovNet

class Title_Module(QWidget):
    def __init__(self):
        super().__init__()
        self.init_UI()
    
    def init_UI(self):
        self.mainlayout = QHBoxLayout()

        self.frame = QFrame()
        # 设置边框形状为 Box，阴影可选
        self.frame.setFrameShape(QFrame.Box)
        self.frame.setFrameShadow(QFrame.Sunken)
        self.frame_layout = QHBoxLayout(self.frame)
      
        self.title = QLabel("鼾声检测系统")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.frame_layout.addWidget(self.title)

        self.mainlayout.addWidget(self.frame)
        self.setLayout(self.mainlayout)
        
class Func_Module(QWidget):
    def __init__(self):
        super().__init__()
        self.init_UI()

    def init_UI(self):
        self.mainlayout = QHBoxLayout()

        self.frame = QFrame()
        # 设置边框形状为 Box，阴影可选
        self.frame.setFrameShape(QFrame.Box)
        self.frame.setFrameShadow(QFrame.Sunken)
        self.frame_layout = QHBoxLayout(self.frame)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.figure = Figure()
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)
        self.ax1.axis("off")
        self.ax2.axis("off")
        self.figure_canvas = FigureCanvas(self.figure)

        self.table = QTableWidget()
        self.table.setRowCount(5)
        self.table.setColumnCount(2)
        header = self.table.horizontalHeader()
        # 让所有列都根据内容大小自动调整
        header.setSectionResizeMode(QHeaderView.Stretch)

        self.left_layout.addWidget(self.figure_canvas, 8)
        self.left_layout.addWidget(self.table, 2)

        # 文件导入区域
        self.file_box = QGroupBox("文件导入")
        self.file_box_layout = QFormLayout()
        self.file_box.setLayout(self.file_box_layout)

        # 图像序列行
        self.figure_label = QLabel("图像序列：")
        self.figure_combobox = QComboBox()
        self.figure_combobox.setPlaceholderText("请选择图像序列文件夹")
        self.figure_combobox.setFixedWidth(400)
        self.figurebtn = QPushButton("选择序列")
        self.figurebtn.clicked.connect(self.open_image_sequences_folder)

        # 创建一个容器，将文本框和按钮水平排列
        figure_field_widget = QWidget()
        figure_field_layout = QHBoxLayout(figure_field_widget)
        figure_field_layout.setContentsMargins(0, 0, 0, 0)
        figure_field_layout.setSpacing(5)
        figure_field_layout.addWidget(self.figure_combobox)
        figure_field_layout.addWidget(self.figurebtn)

        self.file_box_layout.addRow(self.figure_label, figure_field_widget)

        # 视频行
        self.video_label = QLabel("视频序列:")
        self.video_combobox = QComboBox()
        self.video_combobox.setPlaceholderText("请选择视频")
        self.video_combobox.setFixedWidth(400)
        self.videobtn = QPushButton("选择视频")  # 假设视频也是用“选择文件夹”或“选择文件”的操作

        # 同样创建容器
        video_field_widget = QWidget()
        video_field_layout = QHBoxLayout(video_field_widget)
        video_field_layout.setContentsMargins(0, 0, 0, 0)
        video_field_layout.setSpacing(5)
        video_field_layout.addWidget(self.video_combobox)
        video_field_layout.addWidget(self.videobtn)

        self.file_box_layout.addRow(self.video_label, video_field_widget)
   
        # 系统信息区域 —— 动态更新部分
        self.system_group = QGroupBox("系统信息")
        self.system_layout = QVBoxLayout()
        self.system_group.setLayout(self.system_layout)
        # 下面几个标签均为成员变量，用于实时更新
        self.os_label = QLabel()
        self.computer_label = QLabel()
        self.processor_label = QLabel()
        self.cpu_usage_label = QLabel()   # 新增，用于显示 CPU 使用率
        self.memory_label = QLabel()
        self.gpu_label = QLabel()

        # 添加到系统信息布局中
        self.system_layout.addWidget(self.os_label)
        self.system_layout.addWidget(self.computer_label)
        self.system_layout.addWidget(self.processor_label)
        self.system_layout.addWidget(self.cpu_usage_label)
        self.system_layout.addWidget(self.memory_label)
        self.system_layout.addWidget(self.gpu_label)

        self.result_box = QGroupBox("检测结果")
        self.result_box_layout = QGridLayout()
        self.result_box.setLayout(self.result_box_layout)
        self.result_box_layout.setContentsMargins(10, 10, 10, 10)  # 四周边距
        self.result_box_layout.setSpacing(5)                       # 控件间距
        self.result_box_layout.setAlignment(Qt.AlignTop)

        ### CPU or GPU ###
        self.device_layout = QHBoxLayout()
        self.device_layout.setSpacing(100)
        
        self.CPUbtn = QRadioButton()
        self.CPU_label = QLabel("CPU")
        self.CPU_layout = QHBoxLayout()
        self.CPU_layout.setSpacing(5)
        self.CPU_layout.addWidget(self.CPUbtn)
        self.CPU_layout.addWidget(QLabel("CPU"))

        self.GPUbtn = QRadioButton()
        self.GPU_label = QLabel("GPU")
        self.GPU_layout = QHBoxLayout()
        self.GPU_layout.setSpacing(5)
        self.GPU_layout.setContentsMargins(0, 0, 0, 0)
        self.GPU_layout.addWidget(self.GPUbtn)
        self.GPU_layout.addWidget(QLabel("GPU"))
        
        self.init_device()
        self.CPUbtn.toggled.connect(self.update_device)
        self.GPUbtn.toggled.connect(self.update_device)

        self.CUDA_layout = QHBoxLayout()
        self.CUDAbtn = QPushButton("检查GPU是否可用")
        self.CUDAbtn.clicked.connect(self.check_cuda_is_availabel)
        self.CUDA_status = QLabel("")
        self.CUDA_layout.addWidget(self.CUDAbtn)
        self.CUDA_layout.addWidget(self.CUDA_status)

        self.device_layout.addLayout(self.CPU_layout)
        self.device_layout.addLayout(self.GPU_layout)
        # self.device_layout.addWidget(self.check_GPUbtn)

        self.choose_checkpoint_layout = QHBoxLayout()
        self.choose_checkpoint_layout.setSpacing(0)
        self.choose_checkpoint = QComboBox()
        self.choose_checkpoint.setMinimumWidth(200)
        self.choose_checkpoint.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.open_checkpoint_folder_btn = QPushButton("模型文件夹")
        self.open_checkpoint_folder_btn.clicked.connect(self.open_checkpoint_folder)
        self.choose_checkpoint_layout.addWidget(QLabel("选择模型:"))
        self.choose_checkpoint_layout.addWidget(self.choose_checkpoint)
        self.choose_checkpoint_layout.addSpacing(10)
        self.choose_checkpoint_layout.addWidget(self.open_checkpoint_folder_btn)

        self.speed_layout = QHBoxLayout()
        self.speed_layout.addWidget(QLabel("推理速度:"))
        self.inference_speed = QLabel("")
        self.inference_speed.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        self.speed_layout.addWidget(self.inference_speed)
        self.speed_layout.addSpacing(100)
        self.FPS = QLabel("")
        self.FPS.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        self.speed_layout.addWidget(QLabel("FPS:"))
        self.speed_layout.addWidget(self.FPS)

        # 创建 UI 元素
        self.target_count_layout = QHBoxLayout()

        self.target_count_layout.setSpacing(5)
        self.target_count_layout.setContentsMargins(0, 0, 0, 0)
        self.target_count_label = QLabel("目标数量:")
        self.target_count = QLabel("")
        self.target_count.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # 固定大小
        self.target_count.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        self.target_count_layout.addWidget(self.target_count_label, alignment=Qt.AlignLeft)
        self.target_count_layout.addWidget(self.target_count, alignment=Qt.AlignLeft)        

        self.target_position_layout = QGridLayout()
        self.xy_label = QLabel("目标坐标(x,y):")
       
        self.position_xy = QLabel()
        self.position_xy.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        self.position_xy.setWordWrap(True) 

        # 第一行：标题
        self.target_position_layout.addWidget(self.xy_label, 0, 0, 1, 2)

        # 第二行：显示目标坐标
        self.target_position_layout.addWidget(self.position_xy, 1, 0, alignment=Qt.AlignRight)

        # 添加 Spacer，让 QLabel 组件左对齐
        spacer = QSpacerItem(20, 0, QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.target_position_layout.addItem(spacer, 1, 2)

        
        self.time_use_layout = QHBoxLayout()
        self.time_use_label = QLabel("总用时:")
        self.time_use = QLabel("")
        self.time_use.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        self.time_use_layout.addWidget(self.time_use_label)
        self.time_use_layout.addWidget(self.time_use)

        self.result_box_layout.setRowMinimumHeight(0, 10)
        self.result_box_layout.setRowMinimumHeight(1, 30)
        self.result_box_layout.setRowMinimumHeight(2, 30)
        self.result_box_layout.setRowMinimumHeight(3, 30)
        self.result_box_layout.setRowMinimumHeight(4, 60)
        self.result_box_layout.setRowMinimumHeight(5, 30)
        
        self.result_box_layout.addWidget(QLabel("运行设备"), 0, 0)
        self.result_box_layout.addLayout(self.device_layout, 1, 0, alignment=Qt.AlignLeft)
        self.result_box_layout.addLayout(self.CUDA_layout, 1, 1)
        self.result_box_layout.addLayout(self.choose_checkpoint_layout, 2, 0)
        self.result_box_layout.addLayout(self.speed_layout, 3, 0, alignment=Qt.AlignLeft)
        self.result_box_layout.addLayout(self.target_count_layout, 4, 0, alignment=Qt.AlignLeft)
        self.result_box_layout.addLayout(self.target_position_layout, 5, 0)
        self.result_box_layout.addLayout(self.time_use_layout, 6, 0, alignment=Qt.AlignLeft)
  
        self.operate_box = QGroupBox("操作")
        self.operate_box_layout = QHBoxLayout()
        self.operate_box.setLayout(self.operate_box_layout)
        
        self.startbtn = QPushButton("开始检测")
        self.startbtn.clicked.connect(self.detect)
        self.quitbtn = QPushButton("退出系统")
        self.quitbtn.clicked.connect(QCoreApplication.instance().quit)
        self.operate_box_layout.addWidget(self.startbtn)
        self.operate_box_layout.addWidget(self.quitbtn)

        self.right_layout.addWidget(self.file_box, 1)
        self.right_layout.addWidget(self.system_group, 2)
        self.right_layout.addWidget(self.result_box, 4)
        self.right_layout.addWidget(self.operate_box, 2)

        self.frame_layout.addLayout(self.left_layout, 6)
        self.frame_layout.addLayout(self.right_layout, 4)

        self.mainlayout.addWidget(self.frame)
        self.setLayout(self.mainlayout)

        # 启动定时器，实现系统信息的动态更新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_system_info)
        self.timer.start(800)  # 每1000毫秒更新一次

    def update_system_info(self):
        # 更新操作系统等基本信息
        self.os_label.setText(f"操作系统: {platform.system()} {platform.release()} ({platform.version()})")
        self.computer_label.setText(f"计算机名称: {platform.node()}")
        self.processor_label.setText(f"处理器: {platform.processor()}")

        # 获取并显示 CPU 使用率
        cpu_usage = psutil.cpu_percent(interval=None)
        self.cpu_usage_label.setText(f"CPU: 使用率: {cpu_usage}%")

        # 更新内存信息
        mem = psutil.virtual_memory()
        self.memory_label.setText(f"内存: {mem.total/(1024**3):.2f} GB  使用率: {mem.percent}%")
        
        # 更新 GPU 信息
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = "; ".join(
                    [f"{gpu.name} 负载: {gpu.load*100:.1f}%, 显存: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB"
                     for gpu in gpus]
                )
            else:
                gpu_info = "未检测到 GPU 信息"
        except Exception as e:
            gpu_info = "获取 GPU 信息失败"
        self.gpu_label.setText("GPU: " + gpu_info)

    def check_cuda_is_availabel(self):
        if torch.cuda.is_available():
            self.CUDA_status.setText("√")
            self.CUDA_status.setStyleSheet("color: green; font-size: 20px; font-weight: bold;")
        else:
            self.CUDA_status.setText("×")
            self.CUDA_status.setStyleSheet("color: red; font-size: 20px; font-weight: bold;")
            self.GPUbtn.setEnabled(False)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("CUDA不可用")
            msg.setText("检测您的CUDA不可用，请检查您的GPU配置")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def open_checkpoint_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模型文件夹", os.path.join(QDir.currentPath(), "experiment"))
        if folder:
            pattern = os.path.join(folder, "*.pth")
            model_files = glob.glob(pattern)
            if model_files:
                for model_path in model_files:
                    # filename = os.path.basename(mode_path)
                    self.choose_checkpoint.addItem(model_path)
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("模型文件未找到")
                msg.setText("未找到任何模型权重文件，请检查路径")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
    
    def open_image_sequences_folder(self):
        """ 打开文件夹并添加子文件夹到 ComboBox，默认选择第一个 """
        folder = QFileDialog.getExistingDirectory(self, "选择图像序列", os.path.join(QDir.currentPath(), "dataset", "validset_final"))
        
        if folder:
            subfolders = [os.path.join(folder, name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

            self.figure_combobox.clear()  # **清空旧数据**
            
            if subfolders:
                for sequences in subfolders:
                    self.figure_combobox.addItem(sequences)  # **添加文件夹**
                
                # **默认选中第一个文件夹**
                self.figure_combobox.setCurrentIndex(0)  # **显示第一个文件夹**
            
            print(f"添加的文件夹: {subfolders}")  # **调试输出**
    
    def init_device(self):
        # 默认选中 CPU
        if not (self.CPUbtn.isChecked() or self.GPUbtn.isChecked()):
            self.CPUbtn.setChecked(True)  
            self.device = "cpu"

    def update_device(self):
        # 如果 GPU 按钮被选中，设置 self.device 为 "GPU"
        if self.GPUbtn.isChecked():
            self.device = "cuda"
        if self.CPUbtn.isChecked():
            self.device = 'cpu'
        print(self.device)

    def detect(self):
        self.timer = QTimer()  # 创建 QTimer
        self.timer.timeout.connect(self.process_next_image)  
        
        # 检查是否选择设备
        if not (self.CPUbtn.isChecked() or self.GPUbtn.isChecked()):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("设备选择错误")
            msg.setText("未检测到有效的设备选择。请确保在 CPU 与 GPU 之间选择一种作为目标设备!")
            msg.setDefaultButton(QMessageBox.Ok)
            msg.exec_()
            return
        FramesNumber = 2
        
        self.model = CovNet().to(self.device)
        # self.file_list = [for file in os.path.walks]
        with torch.no_grad():
            self.model.eval()
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.img_path = []  
            checkpoint = torch.load(f"{self.choose_checkpoint.currentText()}", map_location="cuda")
            self.model.load_state_dict(checkpoint['m'])

            self.file_list = [files for files in os.listdir(self.figure_combobox.currentText())]
            self.file_list = sorted(self.file_list, key=lambda x: int(x.split('.')[0]))
            self.file_list = [os.path.join(self.figure_combobox.currentText(), i) for i in self.file_list]
            img_list = [self.transform(Image.open(item)) for item in self.file_list]
            img_list = [img_list[i : i+FramesNumber] for i in range(0,len(img_list)-FramesNumber)]
            for item in img_list:
                self.img_path.append(item)
            
            # 设置计时器起始时间
            self.start_time = time.perf_counter()
            self.timer.start(11)
            return

    def process_next_image(self):
        """逐帧推理（模拟视频流）"""
        # start_time = time.perf_counter()
        if self.img_path:
            img = self.img_path.pop(0)  # 取出队列中的第一帧
            img_path = self.file_list.pop(0)
            self.start_inference(img, img_path)
            # 计算当前用时
            elapsed_time = time.perf_counter() - self.start_time
            self.time_use.setText(f"{elapsed_time:.3f}s")

    def start_inference(self, img, img_path):
        """执行推理并更新 UI"""
        start = time.perf_counter()
        res, top5 = self.model(img[0].to(self.device).unsqueeze(dim=0),
                         img[1].to(self.device).unsqueeze(dim=0))
        end = time.perf_counter()

        res_standard = (res - res.mean()) / (res.std() + 1e-8)
        split_image = torch.sigmoid(res_standard.detach()).to("cpu").numpy() * 255
        thread_value = split_image.max() * 0.999
        split_image[split_image < thread_value] = 0
        split_image[split_image > thread_value] = 255
        
         # 目标检测
        target_count, targets = self.find_targets(split_image)

        # 更新 UI：目标数量
        self.target_count.setText(f"{target_count}")

        # 更新 UI：目标坐标
        targets_center = [item['center'] for item in targets]
        self.position_xy.setText(str(targets_center))
        
        self.ax1.imshow(mpimg.imread(img_path))
        self.ax2.imshow(split_image)
        self.ax1.set_title("原始图像", fontsize=14, fontweight='bold', fontname='SimHei')
        self.ax2.set_title("检测分割图", fontsize=14, fontweight='bold', fontname='SimHei')
        self.ax1.axis("off")
        self.ax2.axis("off")
        self.figure_canvas.draw()
        
        inference_time = end - start
        fps = 1.0 / inference_time if inference_time > 0 else 0

        self.inference_speed.setText(f"{inference_time:.3f}s")
        self.FPS.setText(f"{fps:.3f}")

    def find_targets(self, split_image):
        """检测所有目标并返回目标数量和坐标"""
        split_image = split_image.astype(np.uint8)

        # 查找轮廓（目标区域）
        contours, _ = cv2.findContours(split_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        targets = []  # 存储目标信息
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # 获取目标边界框
            center_x = x + w // 2
            center_y = y + h // 2
            targets.append({"bbox": (x, y, w, h), "center": (center_x, center_y)})

        return len(targets), targets  # 返回目标数量 & 目标列表


class Status_Module(QWidget):
    def __init__(self):
        super().__init__()
        self.init_UI()
    
    def init_UI(self):
        return
    
class mainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_UI()

    def init_UI(self):
        self.setWindowIcon(QIcon("J20.webp"))
        self.setGeometry(100,100,1200,800)
        self.setWindowTitle("红外小目标检测软件v1.0")
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.mainlayout = QVBoxLayout(central_widget)
        self.title = Title_Module()
        self.func = Func_Module()
        self.status = Status_Module()

        self.mainlayout.addWidget(self.title)
        self.mainlayout.addWidget(self.func, 9)
        self.mainlayout.addWidget(self.status)


class ISTDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_UI()

    def init_UI(self):
        self.setWindowIcon(QIcon("J20.webp"))
        self.setGeometry(100,100,1200,800)
        self.setWindowTitle("红外小目标检测软件v1.0")
        # self.setMenuBar()
        # 创建主Widget作为centralWidget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
 
        self.mainlayout = QVBoxLayout(central_widget)

        self.top_group_box = QGroupBox()
        self.top_layout = QHBoxLayout(self.top_group_box)
        self.top_group_box.setLayout(self.top_layout)
        self.title_label = QLabel("鼾声检测系统")
        self.title_label.setAlignment(Qt.AlignCenter)  # 居中
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        # 也可设置颜色，如 self.title_label.setStyleSheet("color: blue; font-size: 18px;")
        self.top_layout.addWidget(self.title_label)

        self.middle_group_box = QGroupBox()
        self.middle_layout = QHBoxLayout()
        self.middle_group_box.setLayout(self.middle_layout)

        self.bottom_group_box = QGroupBox()
        self.bottom_layout = QHBoxLayout()
        self.bottom_group_box.setLayout(self.bottom_layout)

        self.middle_left_layout = QVBoxLayout()
        self.middle_right_layout = QVBoxLayout()

        self.middel_right_top_group_box = QGroupBox("文件导入")
        self.middel_right_top_group_box_layout = QVBoxLayout()
        self.middel_right_top_group_box.setLayout(self.middel_right_top_group_box_layout)

        self.middel_right_middle_group_box = QGroupBox("检测结果")
        self.middel_middel_right_middle_group_box_layout = QVBoxLayout()
        self.middel_right_middle_group_box.setLayout(self.middel_middel_right_middle_group_box_layout)

        self.middel_right_bottom_group_box = QGroupBox("操作")
        self.middel_right_bottom_group_box_layout = QHBoxLayout()
        self.middel_right_bottom_group_box.setLayout(self.middel_right_bottom_group_box_layout)

        self.fig = Figure()
        self.fig_canvas = FigureCanvas(self.fig)

        self.table = QTableWidget()
        # self.table.setMinimumSize(200, 200)
        self.table.setMaximumWidth(1200)
        self.table.setRowCount(5)
        self.table.setColumnCount(2)
        header = self.table.horizontalHeader()
        # 让所有列都根据内容大小自动调整
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        self.sequence_label = QLabel()
        self.sequence_label.setText("选择序列：")
        self.sequence_edit = QLineEdit()
        self.sequence_edit.setPlaceholderText("请选择序列")
        self.middel_right_top_group_box_layout.addWidget(self.sequence_label)
        self.middel_right_top_group_box_layout.addWidget(self.sequence_edit)

        self.middle_left_layout.addWidget(self.fig_canvas, 8)
        self.middle_left_layout.addWidget(self.table, 2)
        self.middle_right_layout.addWidget(self.middel_right_top_group_box, 2)
        self.middle_right_layout.addWidget(self.middel_right_middle_group_box, 4)
        self.middle_right_layout.addWidget(self.middel_right_bottom_group_box, 2)

        self.middle_layout.addLayout(self.middle_left_layout, 7)
        self.middle_layout.addLayout(self.middle_right_layout, 3)

        self.mainlayout.addWidget(self.top_group_box)
        self.mainlayout.addWidget(self.middle_group_box, 9)
        self.mainlayout.addWidget(self.bottom_group_box)
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = mainApp()
    window.show()
    sys.exit(app.exec_())