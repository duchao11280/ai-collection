
from datetime import datetime, time, timedelta

def get_shift_type(check_time: datetime):
    """
    Xác định ca làm việc của một thời điểm bất kỳ.
    """
    date = check_time.date()
    night_shift_start = datetime.combine(date, time(20, 0))
    day_shift_start = datetime.combine(date, time(8, 0))
    night_shift_end = datetime.combine(date, time(8, 0))
    
    if check_time.time() >= time(20, 0):
        return f"Ca đêm của ngày {check_time.strftime('%d-%m-%Y')}"
    elif check_time.time() >= time(8, 0):
        return f"Ca ngày của ngày {check_time.strftime('%d-%m-%Y')}"
    else:
        return f"Ca đêm của ngày {(check_time - timedelta(days=1)).strftime('%d-%m-%Y')}"

# Ví dụ sử dụng
check_time = datetime(2025, 3, 12, 21, 30)  # Thời điểm kiểm tra
print(get_shift_type(check_time))
