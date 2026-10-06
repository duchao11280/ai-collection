import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def create_formatted_excel(data, output_filename="output.xlsx"):
    """
    Create a formatted Excel file with the given data.
    
    Args:
        data: List of dictionaries, each containing the data for one row
        output_filename: Name of the output Excel file
    """
    # Create a new workbook and select the active sheet
    wb = openpyxl.Workbook()
    ws = wb.active
    
    # Define styles
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="000080", end_color="000080", fill_type="solid")  # Navy blue
    
    red_font = Font(color="FF0000")
    center_alignment = Alignment(horizontal='center', vertical='center')
    
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Define headers
    headers = ["STT", "Ngày kiểm tra", "VISION", "Ca làm việc", "Số lượng KT", "OK", "NG", "Tỷ lệ NG"]
    
    # Write headers
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_alignment
        cell.border = border
    
    # Write data
    for row_idx, row_data in enumerate(data, 2):
        # STT
        ws.cell(row=row_idx, column=1).value = row_data.get("STT", "")
        
        # Ngày kiểm tra
        ws.cell(row=row_idx, column=2).value = row_data.get("Ngày kiểm tra", "")
        
        # VISION
        ws.cell(row=row_idx, column=3).value = row_data.get("VISION", "")
        
        # Ca làm việc
        ws.cell(row=row_idx, column=4).value = row_data.get("Ca làm việc", "")
        
        # Số lượng KT
        ws.cell(row=row_idx, column=5).value = row_data.get("Số lượng KT", 0)
        
        # OK
        ws.cell(row=row_idx, column=6).value = row_data.get("OK", 0)
        
        # NG
        ws.cell(row=row_idx, column=7).value = row_data.get("NG", 0)
        
        # Tỷ lệ NG
        ty_le_ng_cell = ws.cell(row=row_idx, column=8)
        ty_le_ng = row_data.get("Tỷ lệ NG", 0)
        ty_le_ng_cell.value = f"{ty_le_ng:.2f}%"
        ty_le_ng_cell.font = red_font
        
        # Apply border and center alignment to all cells in the row
        for col_idx in range(1, 9):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.border = border
            cell.alignment = center_alignment
    
    # Add the "TỔNG" row
    total_row_idx = len(data) + 2
    ws.cell(row=total_row_idx, column=1).value = "TỔNG"
    
    # Calculate totals for columns 5, 6, 7, 8
    for col_idx in range(5, 8):
        total = sum(row.get(headers[col_idx-1], 0) for row in data)
        ws.cell(row=total_row_idx, column=col_idx).value = total
    
    # Calculate and format the total Tỷ lệ NG
    total_kt = sum(row.get("Số lượng KT", 0) for row in data)
    total_ng = sum(row.get("NG", 0) for row in data)
    if total_kt > 0:
        total_ty_le_ng = (total_ng / total_kt) * 100
    else:
        total_ty_le_ng = 0
    
    ty_le_ng_total_cell = ws.cell(row=total_row_idx, column=8)
    ty_le_ng_total_cell.value = f"{total_ty_le_ng:.2f}%"
    ty_le_ng_total_cell.font = red_font
    
    # Apply border and center alignment to all cells in the total row
    for col_idx in range(1, 9):
        cell = ws.cell(row=total_row_idx, column=col_idx)
        cell.border = border
        cell.alignment = center_alignment
    
    # Auto-adjust column widths
    for col_idx in range(1, 9):
        column = get_column_letter(col_idx)
        ws.column_dimensions[column].width = max(len(headers[col_idx-1]) + 2, 12)
    
    # Save the workbook
    wb.save(output_filename)
    return output_filename

# Example usage
if __name__ == "__main__":
    # Sample data matching your example
    sample_data = [
        {
            "STT": 1,
            "Ngày kiểm tra": "2025/03/01",
            "VISION": "N",
            "Ca làm việc": 100000,
            "Số lượng KT": 100000,
            "OK": 99999,
            "NG": 1,
            "Tỷ lệ NG": 0.00
        },
        {
            "STT": 1,
            "Ngày kiểm tra": "2025/03/01",
            "VISION": "D",
            "Ca làm việc": 100000,
            "Số lượng KT": 100000,
            "OK": 99990,
            "NG": 10,
            "Tỷ lệ NG": 0.01
        },
        {
            "STT": 2,
            "Ngày kiểm tra": "2025/03/02",
            "VISION": "N",
            "Ca làm việc": 180000,
            "Số lượng KT": 180000,
            "OK": 99990,
            "NG": 80010,
            "Tỷ lệ NG": 8000.00
        },
        {
            "STT": 3,
            "Ngày kiểm tra": "2025/03/03",
            "VISION": "D",
            "Ca làm việc": 160000,
            "Số lượng KT": 160000,
            "OK": 60000,
            "NG": 100000,
            "Tỷ lệ NG": 16700.00
        },
        {
            "STT": 3,
            "Ngày kiểm tra": "2025/03/03",
            "VISION": "N",
            "Ca làm việc": 170000,
            "Số lượng KT": 170000,
            "OK": 60000,
            "NG": 110000,
            "Tỷ lệ NG": 18300.00
        },
        {
            "STT": 4,
            "Ngày kiểm tra": "2025/03/04",
            "VISION": "D",
            "Ca làm việc": 170000,
            "Số lượng KT": 170000,
            "OK": 60000,
            "NG": 110000,
            "Tỷ lệ NG": 18300.00
        }
    ]
    
    output_file = create_formatted_excel(sample_data, "quality_report.xlsx")
    print(f"Excel file created: {output_file}")