import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

# Dữ liệu mẫu từ MongoDB
data = [
        {
            "_id": {
                "$oid": "67cf8ff2e9326770f3419133"
            },
            "created_at": "2025-03-11 01:20:50.534165+00:00",
            "day": "2025/03/11",
            "ng": 11,
            "number_of_objects": 11,
            "shift": "N"
        },
        {
            "_id": {
                "$oid": "67cecae3120eaa3ee9014a3f"
            },
            "created_at": "2025-03-10 11:20:02.914362+00:00",
            "day": "2025/03/10",
            "ng": 1,
            "number_of_objects": 3,
            "ok": 2,
            "shift": "D"
        },
        {
            "_id": {
                "$oid": "67cfaa8cf9f32e69dccb417f"
            },
            "created_at": "2025-03-10 09:20:02.914362+00:00",
            "day": "2025/03/10",
            "ng": 1,
            "number_of_objects": 31,
            "ok": 30,
            "shift": "N"
        }
    ]

# Chuyển đổi dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Chuyển 'day' thành datetime để sắp xếp
df["created_at"] = pd.to_datetime(df["created_at"], format="%Y/%m/%d")

# Sắp xếp theo ngày tăng dần
df = df.sort_values(by="created_at")

# Chuyển lại thành string theo format ban đầu
df["day"] = df["day"].dt.strftime("%Y/%m/%d")

# Mapping tên cột theo yêu cầu
column_mapping = {
    "day": "Ngày kiểm tra VISION",
    "shift": "Ca làm việc",
    "number_of_objects": "Số lượng KT",
    "ok": "OK",
    "ng": "NG"
}

# Đổi tên cột theo mapping
df.rename(columns=column_mapping, inplace=True)



# Tính tỷ lệ NG (%)
df["Tỷ lệ NG"] = df.apply(lambda row: f"{(row['NG'] / row['Số lượng KT']) * 100:.2f}%" if row['Số lượng KT'] > 0 else "0.00%", axis=1)

df["STT"] = ""
unique_days = df["Ngày kiểm tra VISION"].unique()
stt_mapping = {day: i + 1 for i, day in enumerate(unique_days)}
df.loc[df["Ngày kiểm tra VISION"].drop_duplicates().index, "STT"] = df["Ngày kiểm tra VISION"].map(stt_mapping)

# Giữ đúng thứ tự cột như yêu cầu
header_order = ["STT", "Ngày kiểm tra VISION", "Ca làm việc", "Số lượng KT", "OK", "NG", "Tỷ lệ NG"]
df = df[[col for col in header_order if col in df.columns]]

# Thêm dòng tổng cuối bảng
total_row = {
    "STT": "TỔNG",
    "Ngày kiểm tra VISION": "",
    "Ca làm việc": "",
    "Số lượng KT": df["Số lượng KT"].sum(),
    "OK": df["OK"].sum(),
    "NG": df["NG"].sum(),
    "Tỷ lệ NG": f"{(df['NG'].sum() / df['Số lượng KT'].sum()) * 100:.2f}%"
}

df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# Xuất file Excel
file_path = "inspection_results.xlsx"
with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Vision Inspection", index=False)

    # Định dạng file Excel
    wb = writer.book
    ws = writer.sheets["Vision Inspection"]

    # Style cho header
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")

    for col_num, col_name in enumerate(df.columns, start=1):
        col_letter = get_column_letter(col_num)
        ws[f"{col_letter}1"].font = header_font
        ws[f"{col_letter}1"].fill = header_fill
        ws.column_dimensions[col_letter].width = 18  # Tăng chiều rộng cột

    prev_day = None
    start_row = 2 
    for row_num in range(2, len(df) + 1):
        day_value = ws[f"B{row_num}"].value

        if day_value == prev_day:
            ws.merge_cells(start_row=start_row, end_row=row_num, start_column=1, end_column=1)
            ws.merge_cells(start_row=start_row, end_row=row_num, start_column=2, end_column=2)
        else:
            start_row = row_num
            prev_day = day_value

    # Style cho cột "Tỷ lệ NG"
    for row_num in range(2, len(df) + 2):
        cell = ws[f"G{row_num}"]
        cell.font = Font(bold=True, color="FF0000")

    # Định dạng dòng tổng (in đậm)
    total_row_idx = len(df)
    for col_num in range(1, len(df.columns) + 1):
        ws[f"{get_column_letter(col_num)}{total_row_idx+1}"].font = Font(bold=True)
        ws.merge_cells(start_row=total_row_idx+1, end_row=total_row_idx+1, start_column=1, end_column=3)

print(f"✅ File Excel đã tạo thành công: {file_path}")
