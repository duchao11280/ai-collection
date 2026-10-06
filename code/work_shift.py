import yaml
import datetime
import pytz

def get_shift(dt):
    """Xác định ca làm việc dựa trên thời gian.
    
    Ca làm việc được định nghĩa như sau:
    - Ca đêm (D): bắt đầu 20:00 và kết thúc 8:00 sáng hôm sau
    - Ca ngày (N): bắt đầu 8:00 và kết thúc 20:00 cùng ngày
    
    Args:
        dt (datetime): Thời điểm cần kiểm tra
        
    Returns:
        tuple: (ngày của ca làm việc, tên ca làm việc)
    """
    timezone = "Asia/Ho_Chi_Minh"
    print("tz:", dt.tzinfo)
    if timezone:
        try:
            tz = pytz.timezone(timezone)
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            dt = dt.astimezone(tz)
        except Exception as e:
            print(e)
    current_date = dt.date()
    current_hour = dt.hour
    current_datetime = dt

    # Đọc cấu hình từ file YAML

    with open("./shift.yaml", 'r') as file:
        config = yaml.safe_load(file)
        shifts = config.get('shift', {})
    
    current_tz = dt.tzinfo
    # Kiểm tra từng ca trong cấu hình
    for shift_name, shift_hours in shifts.items():
        start_hour = shift_hours.get('start')
        end_hour = shift_hours.get('end')
        
        # Tạo datetime cho thời điểm bắt đầu và kết thúc ca
        start_datetime = datetime.datetime.combine(current_date, datetime.time(start_hour, 0))
        if current_tz is not None:
            start_datetime = current_tz.localize(start_datetime) if hasattr(current_tz, 'localize') else start_datetime.replace(tzinfo=current_tz)
        # Xử lý ca qua đêm
        if start_hour >= end_hour:  # Ca qua đêm (ví dụ: 20:00 - 8:00)
            end_datetime = datetime.datetime.combine(current_date + datetime.timedelta(days=1), datetime.time(end_hour, 0))
            if current_tz is not None:
                end_datetime = current_tz.localize(end_datetime) if hasattr(current_tz, 'localize') else end_datetime.replace(tzinfo=current_tz)
            # Nếu thời điểm hiện tại từ 0:00 đến end_hour, ca làm việc thuộc về ngày hôm trước
            if 0 <= current_hour < end_hour:
                previous_date = current_date - datetime.timedelta(days=1)
                prev_start_datetime = datetime.datetime.combine(previous_date, datetime.time(start_hour, 0))
                prev_end_datetime = datetime.datetime.combine(current_date, datetime.time(end_hour, 0))
                if current_tz is not None:
                    prev_start_datetime = current_tz.localize(prev_start_datetime) if hasattr(current_tz, 'localize') else prev_start_datetime.replace(tzinfo=current_tz)
                    prev_end_datetime = current_tz.localize(prev_end_datetime) if hasattr(current_tz, 'localize') else prev_end_datetime.replace(tzinfo=current_tz)
                if prev_start_datetime <= current_datetime < prev_end_datetime:
                    return previous_date, shift_name
            
            # Nếu thời điểm hiện tại từ start_hour đến 24:00
            elif current_hour >= start_hour:
                if start_datetime <= current_datetime < end_datetime:
                    return current_date, shift_name
         # Ca trong ngày (ví dụ: 8:00 - 20:00)
        else: 
            end_datetime = datetime.datetime.combine(current_date, datetime.time(end_hour, 0))
            if current_tz is not None:
                end_datetime = current_tz.localize(end_datetime) if hasattr(current_tz, 'localize') else end_datetime.replace(tzinfo=current_tz)
            if start_datetime <= current_datetime < end_datetime:
                return current_date, shift_name
    
    # Trường hợp không thuộc ca nào (không xảy ra với cấu hình hiện tại)
    return None, None

# Hàm kiểm tra
def test_shift_detection():
    test_cases = [
        # Ngày 11/3/2025
        (datetime.datetime(2025, 3, 11, 12, 0, tzinfo=pytz.UTC), datetime.date(2025, 3, 11), 'N'),  # 12:00 - Ca ngày của ngày 11
        (datetime.datetime(2025, 3, 11, 21, 0, tzinfo=pytz.UTC), datetime.date(2025, 3, 11), 'D'),  # 21:00 - Ca đêm của ngày 11
        
        # Ngày 12/3/2025
        (datetime.datetime(2025, 3, 12, 7, 0, tzinfo=pytz.UTC), datetime.date(2025, 3, 11), 'D'),   # 7:00 - Vẫn là ca đêm của ngày 11
        (datetime.datetime(2025, 3, 12, 10, 0, tzinfo=pytz.UTC), datetime.date(2025, 3, 12), 'N'),  # 10:00 - Ca ngày của ngày 12
        (datetime.datetime(2025, 3, 12, 22, 0, tzinfo=pytz.UTC), datetime.date(2025, 3, 12), 'D'),  # 22:00 - Ca đêm của ngày 12
        
        # Ngày 13/3/2025
        (datetime.datetime(2025, 3, 13, 6, 0), datetime.date(2025, 3, 12), 'D'),   # 6:00 - Vẫn là ca đêm của ngày 12
    ]
    
    for dt, expected_date, expected_shift in test_cases:
        date, shift = get_shift(dt)
        print(f"Thời gian: {dt}, Ca làm việc: {date}, {shift}")
        assert date == expected_date and shift == expected_shift, \
            f"Lỗi: {dt} trả về {date}, {shift} thay vì {expected_date}, {expected_shift}"
    
    print("Tất cả các kiểm tra đều thành công!")

# Sử dụng ví dụ
if __name__ == "__main__":
    # # # Sử dụng với thời gian hiện tại

    # now = datetime.datetime.now(pytz.utc)
    # date, shift = get_shift(now)
    # print(f"Hiện tại: {now}, Ca làm việc: {date}, {shift}")
    
    # # Hoặc kiểm tra với một thời điểm cụ thể
    specific_time = datetime.datetime(2025, 3, 12, 0, 30, tzinfo=pytz.utc)
    date, shift = get_shift(specific_time)
    print(f"Thời điểm cụ thể: {specific_time}, Ca làm việc: {date}, {shift}")
    
    # # Chạy các kiểm tra
    # test_shift_detection()