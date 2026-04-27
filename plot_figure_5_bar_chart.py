import numpy as np
import matplotlib.pyplot as plt

def plot_figure_5():
    # 统一使用学术规范字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 提取您的数据 (保留两位小数用于展示)
    subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
    
    # Standard Transformer Best Acc (%)
    std_acc = [96.55, 79.31, 100.00, 86.21, 93.10, 75.86, 86.21, 96.55, 89.66]
    
    # MoE (No Transfer / From Scratch) Best Acc (%)
    moe_acc = [96.55, 79.31, 100.00, 89.66, 89.66, 75.86, 89.66, 96.55, 93.10]
    
    # 设置柱状图的宽度和位置
    x = np.arange(len(subjects))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制分组柱状图
    # 蓝色代表标准 Transformer (Baseline)，橙色代表您的创新架构 MoE
    rects1 = ax.bar(x - width/2, std_acc, width, label='Standard Transformer', color='#1f77b4', alpha=0.85)
    rects2 = ax.bar(x + width/2, moe_acc, width, label='MoE Transformer (From Scratch)', color='#ff7f0e', alpha=0.85)
    
    # 添加数值标签的函数
    def autolabel(rects, is_moe=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            
            # 标记需要特别高亮的受试者 (A04: index 3, A07: index 6, A09: index 8)
            highlight_idx = [3, 6, 8]
            
            # 如果是 MoE 且在这个三个受试者上，使用红色粗体显示数值并加上向上箭头
            if is_moe and i in highlight_idx and moe_acc[i] > std_acc[i]:
                ax.annotate(f'↑ {height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, color='#d62728', fontweight='bold')
                
                # 给柱子加上醒目的红色边框
                rect.set_edgecolor('#d62728')
                rect.set_linewidth(2)
            else:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, color='black')

    # 调用标签函数
    autolabel(rects1, is_moe=False)
    autolabel(rects2, is_moe=True)
    
    # 调整图表细节
    ax.set_ylabel('Best Classification Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Subjects (BCIC IV 2a)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 5: Peak Accuracy Comparison Illuminating MoE Performance Ceilings', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=12)
    
    # 设置 y 轴范围，留出顶部空间给数值标签
    ax.set_ylim(60, 105)
    
    # 添加网格线以增强可读性
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # 调整图例
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # 增加辅助说明文本框，解释红色箭头的意义
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#d62728', alpha=0.9)
    ax.text(0.02, 0.95, 'Highlighted (↑):  Effectively\nextends in MoE architecture', 
            transform=ax.transAxes, fontsize=11, color='#d62728', fontweight='bold',
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = 'Figure_5_MoE_Peak_Accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已生成并保存至：{save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_5()