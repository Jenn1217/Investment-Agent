import pandas as pd

# 1. 设置文件路径 (注意路径前的 r 是为了防止转义)
file_path = r'C:\Users\Admin\Desktop\cluster_news\1.12-1.18.csv'

# 2. 读取 CSV 文件
# 注意：中文 CSV 通常使用 'utf-8' 或 'gbk' 编码。
# 如果报错 UnicodeDecodeError，请尝试将 encoding='utf-8' 改为 'gbk' 或 'gb18030'
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 读取失败，尝试使用 GBK 编码...")
    df = pd.read_csv(file_path, encoding='gbk')

# 3. 将 F4003 列转换为日期时间格式
# errors='coerce' 表示如果遇到无法转换的格式，将其设为 NaT (空时间)，避免报错
df['F4003'] = pd.to_datetime(df['F4003'], errors='coerce')

# ---------------------------------------------------------
# 【请在此处修改你需要的时间段】
start_time = '2026-01-18 00:00:00'  # 开始时间
end_time   = '2026-01-18 23:59:59'  # 结束时间
# ---------------------------------------------------------

# 4. 执行筛选
# 逻辑：F4003 大于等于开始时间 且 小于等于结束时间
filtered_data = df[(df['F4003'] >= start_time) & (df['F4003'] <= end_time)]

# 5. 检查结果并保存
if not filtered_data.empty:
    print(f"筛选成功！共找到 {len(filtered_data)} 条数据。")
    
    # 保存为新文件，放在同目录下
    output_path = r'C:\Users\Admin\Desktop\cluster_news\118.csv'
    
    # utf-8-sig 可以防止 Excel 打开中文乱码
    filtered_data.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"文件已保存至: {output_path}")
    
    # 打印前几行看看
    print(filtered_data[['F4003', 'F4020']].head()) 
else:
    print("未在指定时间段内找到数据，请检查时间范围。")