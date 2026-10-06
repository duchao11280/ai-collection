# #3. Data preprocessing

Data preprocessing (tiền xử lý dữ liệu) là bước quan trọng trong quá trình xây dựng các mô hình machine learning. Mục tiêu là chuẩn bị dữ liệu thô sao cho phù hợp để áp dụng các thuật toán machine learning. Quá trình này bao gồm nhiều công đoạn, từ việc làm sạch dữ liệu đến việc biến đổi dữ liệu.

## Data types
Trong machine learning, dữ liệu có thể được phân loại thành nhiều loại khác nhau:

1. **Numerical Data** (Dữ liệu số): Chẳng hạn như chiều cao, cân nặng, tuổi tác. Đây là những dữ liệu có thể đo lường và có thể biểu diễn dưới dạng số.

2. **Categorical Data** (Dữ liệu phân loại): Bao gồm các giá trị thuộc về các nhóm hoặc danh mục, chẳng hạn như màu sắc (đỏ, xanh, vàng) hay quốc gia (Việt Nam, Mỹ, Nhật Bản).

3. **Ordinal Data** (Dữ liệu thứ bậc): Là dạng dữ liệu phân loại nhưng có thứ tự, ví dụ như xếp hạng (tốt, trung bình, kém).

4. **Time Series Data** (Dữ liệu chuỗi thời gian): Dữ liệu được thu thập theo thời gian, với mỗi điểm dữ liệu tương ứng với một thời điểm cụ thể. Dữ liệu chuỗi thời gian thường được sử dụng trong các bài toán dự báo, như dự báo giá cổ phiếu, thời tiết, hoặc các chỉ số kinh tế.

5. **Text Data** (Dữ liệu văn bản): Dữ liệu dạng văn bản thường không có cấu trúc rõ ràng và cần được xử lý đặc biệt trước khi áp dụng các thuật toán machine learning. Các kỹ thuật như tokenization, stemming, hoặc sử dụng các mô hình ngôn ngữ là rất quan trọng để trích xuất thông tin từ văn bản.

6. **Image Data**  (Dữ liệu hình ảnh): Dữ liệu dưới dạng hình ảnh thường được biểu diễn dưới dạng ma trận các giá trị pixel. Đây là loại dữ liệu phổ biến trong các bài toán về nhận diện hình ảnh, phân loại hình ảnh.

7. **Audio Data** (Dữ liệu âm thanh): Dữ liệu âm thanh được biểu diễn dưới dạng sóng âm hoặc chuỗi các mẫu số học. Nó thường được xử lý trong các bài toán như nhận diện giọng nói, phân loại âm thanh.

8. **Video Data** (Dữ liệu video): Video là tập hợp của nhiều khung hình ảnh kết hợp với dữ liệu âm thanh. Việc phân tích video thường đòi hỏi xử lý cả dữ liệu hình ảnh lẫn âm thanh.


## Data cleaning
Dữ liệu thường chứa nhiều lỗi, thiếu sót hoặc không đúng định dạng, do đó cần phải làm sạch để đảm bảo tính chính xác và nhất quán.

> "Một thuật toán hợp lý với dữ liệu tốt sẽ vượt trội hơn một thuật toán tuyệt vời với dữ liệu không tốt."  (Andrew NG, Machine Learning in Production)

### Data quality
Dữ liệu chất lương cao cần đảm bảo các tiêu chí chất lượng bao gồm:

* Tính hợp lệ (Validity):
* Sự chính xác (Accuracy): Dữ liệu phải phản ánh chính xác thực tế. Bất kỳ sai sót nào trong dữ liệu có thể dẫn đến các kết quả không đúng đắn từ mô hình.
Ví dụ: Trong một bộ dữ liệu về bệnh nhân, việc ghi sai tuổi hoặc giới tính có thể dẫn đến kết quả chẩn đoán sai.
* Sự hoàn thiện (Completeness): Dữ liệu không nên thiếu các giá trị quan trọng. Các trường hợp giá trị bị thiếu nên được xử lý một cách hợp lý.
* Sự nhất quán (Consistency): Dữ liệu phải nhất quán về cấu trúc và định dạng. Không nên có sự khác biệt giữa các bản ghi hoặc giữa các nguồn dữ liệu khác nhau. Ví dụ như trong bộ dữ liệu có một cột chứa thông tin ngày tháng, thì toàn bộ cột này phải có định dạng ngày tháng giống nhau.
* Tính duy nhất (Uniqueness): Không nên có sự trùng lặp trong dữ liệu. Mỗi bản ghi nên đại diện cho một đối tượng duy nhất trong tập dữ liệu.
* Tính đồng nhất (Uniformity): Mức độ dữ liệu được chỉ định bằng cách sử dụng cùng một đơn vị đo lường. Ví dụ như tiền tệ thì chỉ định là USD hoặc YEN.

### Denoising
Nhiễu trong dữ liệu có thể làm sai lệch kết quả phân tích. Quá trình loại bỏ nhiễu nhằm mục đích giảm thiểu hoặc loại bỏ các điểm dữ liệu không liên quan hoặc sai lệch.

### Data Scrubbing, Handling missing values
Dữ liệu thường bị thiếu một số giá trị. Có nhiều cách để xử lý:

* **Loại bỏ các dòng dữ liệu thiếu**: Nếu số lượng dòng thiếu không nhiều và không ảnh hưởng lớn đến phân tích.
* **Điền giá trị trung bình/giá trị phổ biến nhất**: Đối với dữ liệu số, có thể điền bằng giá trị trung bình. Đối với dữ liệu phân loại, có thể điền bằng giá trị xuất hiện nhiều nhất.
* **Sử dụng các thuật toán dự đoán**: Áp dụng các mô hình dự đoán để ước lượng giá trị thiếu.

Ví dụ: Với trường hợp thiếu data như:
```python
data = {'Age': [25, 27, 30, 35, 29],
        'Salary': [50000, 54000, None, 58000, 52000]}
```
Ta có thể loại bỏ dữ liệu của tuổi `30` vì salary bị `None`.
## Data transformation
Biến đổi dữ liệu là quá trình thay đổi dữ liệu thô thành dạng phù hợp để phân tích.
### Normalization and Standardization

#### Normalization
Normalization là quá trình đưa dữ liệu về một khoảng giá trị cố định, thường là [0, 1]. Kỹ thuật này thường được áp dụng khi bạn muốn đảm bảo rằng tất cả các feature có tầm quan trọng như nhau trong quá trình học.
$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$
Ví dụ: Một tập dữ liệu có các giá trị tuổi từ 18 đến 60. Sau khi chuẩn hóa, tất cả các giá trị sẽ nằm trong khoảng từ 0 đến 1.

#### Standardization
Standardization là quá trình biến đổi dữ liệu sao cho nó có phân phối chuẩn với trung bình là 0 và độ lệch chuẩn là 1. Kỹ thuật này thường được sử dụng khi dữ liệu có phân phối chuẩn hoặc khi bạn muốn áp dụng các thuật toán nhạy cảm với phân phối dữ liệu như hồi quy tuyến tính, SVM.
$$
z = \frac{x - \mu}{\sigma}
$$
Trong đó $\mu$ là giá trị trung bình và $\sigma$ là độ lệch chuẩn của dữ liệu.
Ví dụ: Tập dữ liệu về chiều cao có giá trị trung bình là 170cm và độ lệch chuẩn là 10cm. Sau khi chuẩn tắc hóa, các giá trị sẽ được biểu diễn theo số độ lệch chuẩn từ giá trị trung bình.

### Encoding Categorical Variables (Mã hóa các biến phân loại)
Các biến phân loại (categorical variables) thường không thể sử dụng trực tiếp trong các mô hình machine learning vì chúng là dữ liệu dạng chuỗi. Việc mã hóa biến phân loại thành các giá trị số là cần thiết.

* One-Hot Encoding: One-Hot Encoding biến các giá trị phân loại thành các vector nhị phân (0 và 1), trong đó mỗi giá trị duy nhất sẽ có một cột riêng.
Ví dụ: Biến phân loại "Color" với các giá trị ['Blue', 'Green', 'Red'] sẽ được mã hóa thành 3 cột: [Red, Green, Blue]. Nếu giá trị là "Red", mã hóa sẽ là [1, 0, 0].
``` python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({'Color': ['Blue', 'Green', 'Red']})
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[['Color']])
print(encoded_data)
```
* Label Encoding biến các giá trị phân loại thành các con số nguyên tương ứng. Kỹ thuật này thường được sử dụng khi các giá trị phân loại có thứ tự. Ví dụ: Biến phân loại "Size" với các giá trị ['Large', 'Medium', 'Small'] có thể được mã hóa thành [0, 1, 2].
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({'Size': ['Large', 'Medium', 'Small', 'Medium']})
# note: Sorting of labels in lexicographic order.
encoder = LabelEncoder()
encoded_data = encoder.fit_transform(df['Size'])
print(encoded_data)
```

### Unbiased estimator (Ước lượng không thiên vị)
Một estimator không thiên vị là một phương pháp hoặc công thức đảm bảo rằng kết quả trung bình của các ước lượng sẽ gần với giá trị thật của tham số mà ta đang ước lượng.

## Data splitting
Chia tách dữ liệu là quá trình phân chia dữ liệu thành các tập con để đào tạo và kiểm tra mô hình.

* Training set (Tập đào tạo): Là tập dữ liệu được sử dụng để huấn luyện mô hình. Mô hình sẽ học các mẫu từ dữ liệu này và điều chỉnh các tham số của nó. Thường chiếm 60-80% tổng số dữ liệu.

* Validation Set (Tập xác thực):  Là tập dữ liệu được sử dụng để điều chỉnh siêu tham số của mô hình và ngăn chặn hiện tượng overfitting. Mô hình không học trực tiếp từ dữ liệu này mà chỉ được kiểm tra và điều chỉnh dựa trên nó. Thường chiếm 10-20% tổng số dữ liệu.
* Test Set (Tập kiểm tra): Là tập dữ liệu được giữ lại để kiểm tra hiệu suất của mô hình sau khi đã huấn luyện và điều chỉnh xong. Đây là bước cuối cùng để đánh giá mô hình trước khi đưa vào sử dụng thực tế. Thường chiếm 10-20% tổng số dữ liệu.

## Sampling
Sampling là quá trình chọn một tập con của dữ liệu từ tổng thể (dataset). Mục đích của sampling có thể khác nhau tùy thuộc vào bài toán và yêu cầu cụ thể, nhưng nhìn chung, các mục đích chính bao gồm:

* Giảm kích thước dữ liệu: Khi bạn có một tập dữ liệu rất lớn, việc sử dụng toàn bộ dữ liệu có thể không khả thi về mặt tính toán. Sampling giúp giảm kích thước dữ liệu, làm cho việc huấn luyện mô hình trở nên nhanh chóng và khả thi hơn.
* Cân bằng dữ liệu: Trong trường hợp dữ liệu có sự chênh lệch lớn giữa các lớp (class imbalance), sampling có thể được sử dụng để cân bằng số lượng mẫu giữa các lớp.
* Đại diện cho tổng thể: Khi không thể sử dụng toàn bộ dữ liệu, sampling giúp chọn ra một tập con đại diện để mô hình hóa hoặc phân tích.

Các phương pháp Sampling phổ biến:

* Random Sampling (Lấy mẫu ngẫu nhiên): Mỗi phần tử trong tập dữ liệu đều có cơ hội được chọn như nhau. Thường sử dụng khi dữ liệu đồng nhất và không có cấu trúc phức tạp. Phù hợp khi bạn muốn lấy mẫu đại diện đơn giản và không quan tâm đến các yếu tố khác.
* Stratified Sampling (Lấy mẫu phân tầng): Dữ liệu được chia thành các nhóm nhỏ (tầng), và lấy mẫu từ mỗi nhóm theo tỷ lệ nhất định. Sử dụng khi dữ liệu có sự phân loại rõ ràng (ví dụ: giới tính, độ tuổi) và bạn muốn đảm bảo rằng mỗi tầng được đại diện chính xác trong tập mẫu.
* Systematic Sampling (Lấy mẫu hệ thống): Lấy mẫu theo một khoảng cách đều đặn. Ví dụ, bạn có thể chọn mỗi phần tử thứ k trong tổng thể. Thường được sử dụng khi dữ liệu được sắp xếp hoặc có một trật tự tự nhiên. Phương pháp này dễ thực hiện và có thể hiệu quả nếu tổng thể không có cấu trúc lặp lại.
* Cluster Sampling (Lấy mẫu cụm): Tổng thể được chia thành các cụm (clusters), sau đó một số cụm được chọn ngẫu nhiên và tất cả các phần tử trong cụm đó được lấy mẫu. Phù hợp khi tổng thể rất lớn và được chia thành các cụm tự nhiên (như các khu vực địa lý hoặc tổ chức). Phương pháp này giúp giảm chi phí và thời gian lấy mẫu.
* Oversampling (Lấy mẫu quá mức): Tăng số lượng mẫu từ lớp thiểu số bằng cách tạo bản sao hoặc tạo dữ liệu tổng hợp (synthetic). Sử dụng khi bạn có dữ liệu mất cân bằng với lớp thiểu số rất nhỏ so với lớp đa số. Phương pháp này giúp mô hình học tốt hơn từ lớp thiểu số.
* Undersampling (Lấy mẫu dưới mức): Giảm số lượng mẫu từ lớp đa số để cân bằng với lớp thiểu số. Sử dụng khi bạn có dữ liệu mất cân bằng và muốn giảm bớt lớp đa số để mô hình không bị "unbiased".
* Reservoir Sampling (Lấy mẫu bể chứa): Phương pháp này sử dụng để lấy mẫu ngẫu nhiên từ một luồng dữ liệu (streaming data) mà không biết trước kích thước tổng thể. Mỗi mẫu trong dòng dữ liệu có cơ hội được chọn vào bể chứa với xác suất ngang nhau. Sử dụng khi bạn xử lý dữ liệu lớn hoặc luồng dữ liệu liên tục mà không thể lưu trữ toàn bộ dữ liệu trong bộ nhớ. Phương pháp này giúp chọn ra một mẫu đại diện mà không cần phải lưu trữ toàn bộ dữ liệu.
## Reporting vs BI vs Analytics

## Dimensionality & Numerosity Reduction
### Dimensional Reduction: 
Là một kỹ thuật được sử dụng để có được biểu diễn thu nhỏ hoặc nén của dữ liệu gốc. Nó được chia thành hai thành phần:

* Feature selection: là quá trình loại bỏ các tính năng không liên quan hoặc thừa
* Feature extraction: là quá trình chuyển đổi dữ liệu thành các tính năng phù hợp để lập mô hình.

### Numerosity Reduction
Là một kỹ thuật giảm dữ liệu được sử dụng để giảm khối lượng dữ liệu bằng cách sử dụng các hình thức biểu diễn dữ liệu phù hợp
### Binning sparse value
Là một kỹ thuật để giảm số lượng giá trị bằng cách gom các giá trị tương tự vào một nhóm (bin).

## Equalize Image
Histogram Equalization (HE) là một phương pháp biến đổi cường độ của ảnh sao cho histogram của ảnh đầu ra phân bố đều hơn, giúp tăng cường độ tương phản.
Ưu điểm:
        Hiệu quả trong việc tăng độ tương phản cho ảnh mờ, ảnh có histogram bị dồn về một phía.
Nhược điểm:
        Không hoạt động tốt với ảnh có vùng sáng/tối cục bộ lớn vì áp dụng cùng một phép biến đổi cho toàn bộ ảnh.
        Dễ làm mất chi tiết trong các vùng có độ sáng gần nhau.
        Có nhiều nhiễu trong equalized image vì nó kéo dài histogram.


Adaptive Histogram Equalization (AHE) là phiên bản cải tiến của HE, trong đó ảnh được chia thành các vùng nhỏ (tiles), sau đó HE được áp dụng cho từng vùng thay vì toàn bộ ảnh. Sử dụng nội suy (interpolation) giữa các vùng để tránh hiệu ứng gián đoạn. 
Ưu điểm:
        Cải thiện độ tương phản cục bộ tốt hơn so với HE.
        Giữ lại chi tiết trong ảnh tốt hơn, đặc biệt là trong các ảnh có vùng sáng và tối xen kẽ nhau.
Nhược điểm:
        Có thể gây tăng nhiễu quá mức trong các vùng có nhiễu cao.

CLAHE (Contrast Limited AHE) là một cải tiến của AHE giúp giảm hiện tượng tăng nhiễu quá mức bằng cách đặt giới hạn tương phản khi thực hiện cân bằng histogram trong từng vùng.
Ưu điểm:
        Tránh bị tăng nhiễu quá mức.
        Hiệu quả hơn trong ảnh y tế, ảnh chụp đêm, ảnh có độ nhiễu cao.

## References
* https://towardsdatascience.com/the-ultimate-guide-to-data-cleaning-3969843991d4
* https://en.wikipedia.org/wiki/Extract,_transform,_load
* https://www.linkedin.com/pulse/types-sampling-machine-learning-chirag-subramanian-hnsoc/