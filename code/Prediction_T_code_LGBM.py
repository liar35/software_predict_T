# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 23:38:30 2024

@author: ad
"""
import threading
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas
import webbrowser
import os
import lightgbm as lgb
import sys
from tkinter import ttk

# 检查模型文件路径是否存在
model_paths = {
    'CH4': '../model/CH4_model.json',
    'C2H6': '../model/C2H6_model.json',
    'CO2': '../model/CO2_model.json',
    'H2S': '../model/H2S_model.json'
}

# 检查每个模型文件是否存在
for gas, path in model_paths.items():
    if not os.path.exists(path):
        messagebox.showerror("Error", f"Model file not found: {path}")
        sys.exit()  # 强制退出程序

# 加载预训练的LightGBM模型列表
lgb_models = {}
for gas, path in model_paths.items():
    model = lgb.Booster(model_file=path)
    lgb_models[gas] = model

print("All models loaded successfully.")

# 创建主窗口
root = tk.Tk()
root.title("MOF Gas Breakthrough Time Predictor")
root.geometry("900x600")  # 调整主界面大小

# 启用高 DPI 支持，避免模糊
root.tk.call('tk', 'scaling', 1.25)  # 根据屏幕情况调整比例

# 配色方案
bg_color = "#F0F4F8"  # 非常浅的蓝灰色背景
frame_bg = "#F2F2F2"  # 白色框架背景
text_color = "#333333"  # 深灰色文字
btn_color = "#2980B9"  # 深蓝色按钮
entry_bg = "#FFFFFF"  # 非常浅灰背景
highlight_color = "#E67E22"  # 橙黄色

root.configure(bg=bg_color)  # 设置主窗口背景颜色

# 创建圆角矩形函数
def create_rounded_rectangle(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    points = [x1+radius, y1,
              x1+radius, y1,
              x2-radius, y1,
              x2-radius, y1,
              x2, y1,
              x2, y1+radius,
              x2, y1+radius,
              x2, y2-radius,
              x2, y2-radius,
              x2, y2,
              x2-radius, y2,
              x2-radius, y2,
              x1+radius, y2,
              x1+radius, y2,
              x1, y2,
              x1, y2-radius,
              x1, y2-radius,
              x1, y1+radius,
              x1, y1+radius,
              x1, y1]
    return canvas.create_polygon(points, **kwargs, smooth=True)

# 标题
title_label = tk.Label(root, text="MOF Gas Breakthrough Time Predictor", 
                       font=("Times New Roman", 20, "bold"), bg=bg_color, fg=highlight_color)
title_label.pack(pady=10)  # 使用 pack 布局使标题自动居中

# 在右下角添加人名
names_label = tk.Label(root, text="Author: Zhiwei Qiao, Jinfeng Li, Guangzhou University", font=("Times New Roman", 10), bg=bg_color, fg=text_color)
names_label.place(relx=0.93, rely=0.98, anchor="se")  # 将人名显示在右下角

# Tooltip 按钮
def cmx1():
    window = tk.Tk()
    window.title('Warm prompt')
    window.geometry('350x250')
    link = tk.Label(window, text='The physical properties of gases \nare known from the literature:\nhttps://doi.org/10.1016/j.ces.2024.120470',
                    font=('Arial', 11), anchor="center", fg=highlight_color, bg="#f0f0f0")
    link.place(x=30, y=40)

    def open_url(event):
        webbrowser.open("https://doi.org/10.1016/j.ces.2024.120470", new=0)

    link.bind("<Button-1>", open_url)

btn_tooltip = tk.Button(root, text='Tooltip', font=("Times New Roman", 11), command=cmx1, bg=btn_color, fg='white', bd=0, relief=tk.FLAT)
btn_tooltip.place(x=780, y=40)

# 输入特征框架的Canvas，用于绘制圆角矩形
input_canvas = Canvas(root, bg=bg_color, highlightthickness=0)
input_canvas.place(x=50, y=70, width=790, height=220)
create_rounded_rectangle(input_canvas, 5, 5, 785, 215, radius=20, outline=highlight_color, fill=frame_bg)

# 在日志框架中绘制圆角的白色背景框
log_inner_canvas = Canvas(root, bg=bg_color, highlightthickness=0)
log_inner_canvas.place(x=442, y=292, width=403, height=222)  # 调整边距和大小，避免被遮盖
create_rounded_rectangle(log_inner_canvas, 0, 0, 393, 212, radius=20, outline=highlight_color, fill=entry_bg)

# 在白色圆角矩形内放置文本框，用于显示日志信息
log_text = tk.Text(log_inner_canvas, wrap="word", bg=entry_bg, fg=text_color, bd=0)
log_text.place(x=8, y=8, width=362, height=184)  # 调整文本框的边距和大小，避免遮盖边框

# 日志信息列表存储
log_messages = {
    "predict": [
        "Log Entry 1: Prediction started.",
        "Log Entry 2: Processing input data...",
        "Log Entry 3: Prediction in progress...",
        "Log Entry 4: Prediction complete."
    ],
    "batch_predict": [
        "Log Entry 1: Batch prediction started.",
        "Log Entry 2: Loading file...",
        "Log Entry 3: Processing data...",
        "Log Entry 4: Batch prediction complete."
    ]
}

# 当前日志索引
current_log_index = 0
current_logs = []

# 定义一个函数用于逐步插入日志
def update_log():
    global current_log_index, current_logs
    
    if current_log_index < len(current_logs):
        # 插入当前的日志信息
        log_text.insert(tk.END, f"{current_logs[current_log_index]}\n")
        # 自动滚动到最底部
        log_text.see(tk.END)
        # 更新日志索引
        current_log_index += 1
        # 设置下一条日志的显示时间
        root.after(1000, update_log)  # 每隔1000毫秒（1秒）显示下一条

# 创建独立线程来执行预测任务
def run_threaded_task(task):
    task_thread = threading.Thread(target=task)
    task_thread.start()

# 定义按钮触发的函数
def start_prediction():
    global current_log_index, current_logs
    # 重置日志
    log_text.delete(1.0, tk.END)
    current_log_index = 0
    current_logs = log_messages["predict"]
    run_threaded_task(predict_task)  # 启动预测任务
    update_log()  # 开始显示日志

def start_batch_prediction():
    global current_log_index, current_logs
    # 重置日志
    log_text.delete(1.0, tk.END)
    current_log_index = 0
    current_logs = log_messages["batch_predict"]
    run_threaded_task(batch_predict_task)  # 启动批量预测任务
    update_log()  # 开始显示日志

# 输入验证，确保只允许浮点数输入
def validate_float(P):
    if P == "":
        return True
    try:
        float(P)
        return True
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid number.")
        return False

validate_float_cmd = root.register(validate_float)

# 创建输入框架中的输入项和标签
input_frame = tk.Frame(input_canvas, bg=frame_bg)
input_frame.place(x=10, y=10, width=770, height=200)

label_CH4_ads_eq = tk.Label(input_frame, font=("Times New Roman", 11), text='CH\u2084 Equilibrium Adsorption (mol/kg):', bg=frame_bg, fg=text_color)
label_CH4_ads_eq.grid(row=0, column=0, padx=10, pady=10, sticky="w")
entry_CH4_ads_eq = tk.Entry(input_frame, font=("Times New Roman", 11), width=13, bg=entry_bg, fg=text_color, validate="key", validatecommand=(validate_float_cmd, '%P'))
entry_CH4_ads_eq.grid(row=0, column=1, padx=10)

label_C2H6_ads_eq = tk.Label(input_frame, font=("Times New Roman", 11), text='C\u2082H\u2086 Equilibrium Adsorption (mol/kg):', bg=frame_bg, fg=text_color)
label_C2H6_ads_eq.grid(row=1, column=0, padx=10, pady=10, sticky="w")
entry_C2H6_ads_eq = tk.Entry(input_frame, font=("Times New Roman", 11), width=13, bg=entry_bg, fg=text_color, validate="key", validatecommand=(validate_float_cmd, '%P'))
entry_C2H6_ads_eq.grid(row=1, column=1, padx=10)

label_CO2_ads_eq = tk.Label(input_frame, font=("Times New Roman", 11), text='CO\u2082 Equilibrium Adsorption (mol/kg):', bg=frame_bg, fg=text_color)
label_CO2_ads_eq.grid(row=2, column=0, padx=10, pady=10, sticky="w")
entry_CO2_ads_eq = tk.Entry(input_frame, font=("Times New Roman", 11), width=13, bg=entry_bg, fg=text_color, validate="key", validatecommand=(validate_float_cmd, '%P'))
entry_CO2_ads_eq.grid(row=2, column=1, padx=10)

label_H2S_ads_eq = tk.Label(input_frame, font=("Times New Roman", 11), text='H\u2082S Equilibrium Adsorption (mol/kg):', bg=frame_bg, fg=text_color)
label_H2S_ads_eq.grid(row=3, column=0, padx=10, pady=10, sticky="w")
entry_H2S_ads_eq = tk.Entry(input_frame, font=("Times New Roman", 11), width=13, bg=entry_bg, fg=text_color, validate="key", validatecommand=(validate_float_cmd, '%P'))
entry_H2S_ads_eq.grid(row=3, column=1, padx=10)

label_CH4_ads_sat = tk.Label(input_frame, font=("Times New Roman", 11), text='CH\u2084 Saturation Adsorption (mol/kg):', bg=frame_bg, fg=text_color)
label_CH4_ads_sat.grid(row=0, column=2, padx=10, pady=10, sticky="w")
entry_CH4_ads_sat = tk.Entry(input_frame, font=("Times New Roman", 11), width=13, bg=entry_bg, fg=text_color, validate="key", validatecommand=(validate_float_cmd, '%P'))
entry_CH4_ads_sat.grid(row=0, column=3, padx=10)

label_C2H6_ads_sat = tk.Label(input_frame, font=("Times New Roman", 11), text='C\u2082H\u2086 Saturation Adsorption (mol/kg):', bg=frame_bg, fg=text_color)
label_C2H6_ads_sat.grid(row=1, column=2, padx=10, pady=10, sticky="w")
entry_C2H6_ads_sat = tk.Entry(input_frame, font=("Times New Roman", 11), width=13, bg=entry_bg, fg=text_color, validate="key", validatecommand=(validate_float_cmd, '%P'))
entry_C2H6_ads_sat.grid(row=1, column=3, padx=10)

label_CO2_ads_sat = tk.Label(input_frame, font=("Times New Roman", 11), text='CO\u2082 Saturation Adsorption (mol/kg):', bg=frame_bg, fg=text_color)
label_CO2_ads_sat.grid(row=2, column=2, padx=10, pady=10, sticky="w")
entry_CO2_ads_sat = tk.Entry(input_frame, font=("Times New Roman", 11), width=13, bg=entry_bg, fg=text_color, validate="key", validatecommand=(validate_float_cmd, '%P'))
entry_CO2_ads_sat.grid(row=2, column=3, padx=10)

label_H2S_ads_sat = tk.Label(input_frame, font=("Times New Roman", 11), text='H\u2082S Saturation Adsorption (mol/kg):', bg=frame_bg, fg=text_color)
label_H2S_ads_sat.grid(row=3, column=2, padx=10, pady=10, sticky="w")
entry_H2S_ads_sat = tk.Entry(input_frame, font=("Times New Roman", 11), width=13, bg=entry_bg, fg=text_color, validate="key", validatecommand=(validate_float_cmd, '%P'))
entry_H2S_ads_sat.grid(row=3, column=3, padx=10)

# 显示结果的区域，用于显示预测的结果
results_canvas = Canvas(root, bg=bg_color, highlightthickness=0)
results_canvas.place(x=50, y=287, width=390, height=225)
create_rounded_rectangle(results_canvas, 5, 5, 385, 217, radius=20, outline=highlight_color, fill=frame_bg)

results_frame = tk.Frame(results_canvas, bg=frame_bg)
results_frame.place(x=10, y=10, width=370, height=200)

def predict_task():
    try:
        CH4_ads_eq = float(entry_CH4_ads_eq.get())
        C2H6_ads_eq = float(entry_C2H6_ads_eq.get())
        CO2_ads_eq = float(entry_CO2_ads_eq.get())
        H2S_ads_eq = float(entry_H2S_ads_eq.get())
        CH4_ads_sat = float(entry_CH4_ads_sat.get())
        C2H6_ads_sat = float(entry_C2H6_ads_sat.get())
        CO2_ads_sat = float(entry_CO2_ads_sat.get())
        H2S_ads_sat = float(entry_H2S_ads_sat.get())
    except ValueError:
        messagebox.showerror("Error", "Please input valid float numbers.")
        return

    # 生成输入特征数组
    input_features = np.array([[CH4_ads_eq, C2H6_ads_eq, CO2_ads_eq, H2S_ads_eq,
                                CH4_ads_sat, C2H6_ads_sat, CO2_ads_sat, H2S_ads_sat]])

    # 分别使用加载的模型进行预测
    ch4_pred = lgb_models['CH4'].predict(input_features)
    c2h6_pred = lgb_models['C2H6'].predict(input_features)
    co2_pred = lgb_models['CO2'].predict(input_features)
    h2s_pred = lgb_models['H2S'].predict(input_features)

    # 应用10^x进行log10的反向变换
    ch4_pred = 10 ** ch4_pred
    c2h6_pred = 10 ** c2h6_pred
    co2_pred = 10 ** co2_pred
    h2s_pred = 10 ** h2s_pred

    # 显示预测的突破时间
    result_label_CH4.config(text=f'CH\u2084 Breakthrough Time: {ch4_pred[0]:.2f} h')
    result_label_C2H6.config(text=f'C\u2082H\u2086 Breakthrough Time: {c2h6_pred[0]:.2f} h')
    result_label_CO2.config(text=f'CO\u2082 Breakthrough Time: {co2_pred[0]:.2f} h')
    result_label_H2S.config(text=f'H\u2082S Breakthrough Time: {h2s_pred[0]:.2f} h')

result_label_CH4 = tk.Label(results_frame, font=("Times New Roman", 12), text="CH\u2084 Breakthrough Time: -- h", bg=frame_bg, fg=text_color)
result_label_CH4.grid(row=0, column=0, pady=10)

result_label_C2H6 = tk.Label(results_frame, font=("Times New Roman", 12), text="C\u2082H\u2086 Breakthrough Time: -- h", bg=frame_bg, fg=text_color)
result_label_C2H6.grid(row=1, column=0, pady=10)

result_label_CO2 = tk.Label(results_frame, font=("Times New Roman", 12), text="CO\u2082 Breakthrough Time: -- h", bg=frame_bg, fg=text_color)
result_label_CO2.grid(row=2, column=0, pady=10)

result_label_H2S = tk.Label(results_frame, font=("Times New Roman", 12), text="H\u2082S Breakthrough Time: -- h", bg=frame_bg, fg=text_color)
result_label_H2S.grid(row=3, column=0, pady=10)

# 预测按钮，关联到values函数
btn_predict = tk.Button(root, font=("Times New Roman", 11), text='Predict Breakthrough Time', bg=btn_color, fg='white', command=start_prediction)
btn_predict.place(x=55, y=520)

# 批量计算功能
def open_file():
    filename = filedialog.askopenfilename(title='Open Excel File', filetypes=[("Excel files", "*.xlsx")])
    entry_filename.delete(0, "end")
    entry_filename.insert(0, filename)

def batch_predict_task():
    # 获取文件路径
    file_path = entry_filename.get()
    
    if not file_path:
        messagebox.showerror("Error", "Please select a file!")
        return

    # 读取Excel文件
    try:
        data = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file: {e}")
        return

    # 检查必要的列是否存在
    input_columns = ['CH4_ads_eq', 'C2H6_ads_eq', 'CO2_ads_eq', 'H2S_ads_eq', 
                     'CH4_ads_sat', 'C2H6_ads_sat', 'CO2_ads_sat', 'H2S_ads_sat']

    if not all(col in data.columns for col in input_columns):
        messagebox.showerror("Error", "Input file does not contain the required columns!")
        return

    # 提取输入特征
    input_features = data[input_columns].values

    # 初始化结果存储数组
    lgb_preds = np.zeros((input_features.shape[0], 4))

    # 批量预测：使用对应模型进行预测
    lgb_preds[:, 0] = lgb_models['CH4'].predict(input_features)  # CH4预测
    lgb_preds[:, 1] = lgb_models['C2H6'].predict(input_features)  # C2H6预测
    lgb_preds[:, 2] = lgb_models['CO2'].predict(input_features)   # CO2预测
    lgb_preds[:, 3] = lgb_models['H2S'].predict(input_features)   # H2S预测

    # 应用10^x进行log10的反向变换
    lgb_preds = 10 ** lgb_preds

    # 保存预测结果
    save_predictions(data, lgb_preds)

# 将预测结果保存到Excel文件
def save_predictions(data, lgb_preds):
    # 创建保存结果的文件夹路径和文件名
    save_folder = "../Result"  # 保存的文件夹路径
    output_file = os.path.join(save_folder, "Batch_Predicted_T.xlsx")  # 保存的文件名

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    try:
        # 创建一个DataFrame存储预测结果
        predictions_df = pd.DataFrame(lgb_preds, columns=['CH4_Breakthrough (h)', 'C2H6_Breakthrough (h)', 'CO2_Breakthrough (h)', 'H2S_Breakthrough (h)'])
        
        # 合并原始数据和预测结果
        output_df = pd.concat([data, predictions_df], axis=1)
        
        # 将结果保存到Excel文件
        output_df.to_excel(output_file, index=False)
        
        # 显示成功消息
        messagebox.showinfo("Success", f"Predicted results saved to {output_file}")
    
    except Exception as e:
        # 如果出现异常，显示错误消息
        messagebox.showerror("Error", f"Failed to save file: {str(e)}")

# 批量导入与计算按钮
entry_filename = tk.Entry(root, font=("Times New Roman", 11), width=20, bg=entry_bg)
entry_filename.place(x=480, y=522, height=26)

btn_import = tk.Button(root, text="Import File", font=("Times New Roman", 11), command=open_file, bg=btn_color, fg='white')
btn_import.place(x=618, y=520, width=120)

btn_batch_predict = tk.Button(root, font=("Times New Roman", 11), text='Batch Predict', command=start_batch_prediction, bg=btn_color, fg='white')
btn_batch_predict.place(x=740, y=520)

# 启动主循环
root.mainloop()

