# #2. Mathematics for machine learning

## Linear Algebra

### Vector
Về mặt hình học, một vector thường bao gồm các tọa độ trong không gian n chiều (n chỉ số lượng chiều). Nó biểu thị vị trí của một tọa độ trong không gian n chiều. Theo Merriam-Webster, `vector` là một đại lượng có độ lớn và có hướng, thường được biểu diễn bằng một đoạn thẳng có hướng mà độ dài của nó biểu diễn độ lớn và hướng trong không gian.[^1] 

Còn theo một tài liệu của Cornell University Department of Mathematics, một `vector` chỉ đơn giản là một danh sách có thứ tự các số, được gọi là tọa độ. Ví dụ (2, 5) là vector 2 chiều (2-vector), còn (-3, 1.4, 5) là vector 3 chiều.[^2]

Các loại vector:

* Vector đơn vị (unit vector): có độ lớn là 1
* Vector không (zero vector) (null): có độ lớn bằng 0 và không hướng.
* Vector vị trí (position vector): biểu thị vị trí của một điểm so với một điểm tham chiếu tùy ý.
* Vector kết quả (resultant vector): là tổng của hai hoặc nhiều vector
* Vector bằng nhau (equal vector): Hai hoặc nhiều vector có cùng độ lớn và có cùng hướng.
* Vector cột (column vector): là khi các thành phần của vector được hiển thị theo chiều dọc.

### Matrics

Ma trận là một dãy số hình chữ nhật được xếp theo hàng hoặc theo cột. 

$$ A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} $$

Một số phép tính liên quan đến ma trận như:

* Cộng: Thêm các phần tử tương ứng.

$$ 
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix} 
$$

* Nhân (vô hướng): Nhân một ma trận với một số.


$$ 2 \cdot \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} =  \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix} $$

* Nhân (có hướng): Để nhân 2 ma trận, ta cần số cột ma trận bên trái bằng số hàng của ma trận bên phải. Ví dụ, ta có 2 ma trận:

* Ma trận **A** có kích thước $m \times n$ (m hàng và n cột)
* Ma trận **B** có kích thước $n \times p$ (n hàng và p cột)
* Kết quả của phép nhân $A \times B$ sẽ là ma trận **C** có kích thước $m \times p$

Cách tính phần tử của ma trận kết quả: Phần tử $C_{ij}$ (ở hàng i, cột j) của ma trận **C** được tính bằng tổng tích các phần tử tương ứng từ hàng i của ma trận **A** và cột j của ma trận **B**:

$$ C_{ij} =  \sum_{k=1}^n A_{ik} \cdot B_{kj} $$

Cụ thể hơn, ta có:

$$ 
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, 
B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

Để tính vị trí $C_{11}$ của ma trận **C**:
$$
C_{11} = (1 \cdot 5) + (2 \cdot 7) = 5 + 14 = 19
$$
Tương tự với các phần tử khác, ta được ma trận **C**:

$$
C = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

### Eigenvalues and Eigenvectors

Với 1 ma trận **A**, nếu có 1 vector khác vector không **v** sao cho **Av** = **λv**, thì **v** là vector riêng (eigenvector), còn λ là giá trị riêng (eigenvalue) tương ứng.


## Probability and statistics

Phân phối xác xuất mô tả cách xác suất được phân bố cho các giá trị khác nhau của một biến ngẫu nhiên.

### Biến ngẫu nhiên (random variable)
Biến ngẫu nhiên là một đại lượng có thể nhận các giá trị khác nhau do kết quả của một hiện tượng ngẫu nhiên.
Có 2 loại biến ngẫu nhiên:

* Biến ngẫu nhiên rời rạc (Discrete Random Variable): có thể nhận một số lượng hữu hạn hoặc đếm được các giá trị. Ví dụ: Số lần xuất hiện mặt ngửa khi tung một đồng xu 3 lần (có thể nhận các giá trị 0,1,2 hoặc 3)

* Biến ngẫu nhiên liên tục (Continuous Random Variable): có thể nhận bất kỳ giá trị nào trong một khoảng liên tục. Ví dụ: Chiều cao của một người, thời gian chờ đợi ở trạm xe.

### Phân phối xác suất 
Theo Investopedia, Phân phôi xác suất là một hàm thống kê mô tả tất cả các giá trị và khả năng có thể xảy ra mà một biến ngẫu nhiên có thể đạt được trong một phạm vi nhất định.[^5]

Hàm phân phối xác suất (Probability Mass Function - PMF) cho biến ngẫu nhiên rời rạc: cung cấp xác suất của mỗi giá trị có thể của biến ngẫu nhiên rời rạc.
$$
P(X=x) = p(x)
$$
Hàm mật độ xác suất (Probability Density Function - PDF) cho biến ngẫu nhiên liên tục: cung cấp xác suất rằng biến ngẫu nhiên liên tục nhận một giá trị trong một khoảng cụ thể. 
$$f(x)$$

Các loại Phân phối xác suất:

1. Phân phối nhị thức (Binomial Distribution): 
Là một phân một phân phối xác suất rời rạc với hai tham số $n$ và $p$, ký hiệu của số lượng phép thử thành công trong n phép thử độc lập tìm kết quả có hay không thành công.
$$
P(X=k) = C(n,k) \cdot p^k \cdot (1 - p)^{n-k}
$$
Một ví dụ là Tung đồng xu 100 lần và tính tỷ lệ sấp hoặc ngửa.

2. Phân phối Bernoulli: là một dạng đặc biệt của *phân phối nhị thức* với $n =1$.
3. Phân phối chuẩn (Normal Distribution):
Hay còn được gọi là phân phối Gauss, phân phối này được đặc trưng đầy đủ bởi giá trị trung bình $\mu$ và độ lệch chuẩn $\sigma$ của nó.
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^-{\frac{(x-\mu)^2}{2\sigma^2}}
$$
Phân phối chuẩn có tính đối xứng.
$$
P(X=k) = \frac{\lambda^k e^-k}{k!}
$$
4. Phân phối Poisson (Poisson Distribution):
Là phân phối xác suất rời rạc mô hình hóa số lượng sự kiện xảy ra trong một khoảng thời gian hoặc không gian cố định. Các sự kiện này phải xảy ra độc lập với nhau và tỷ lệ trung bình (số lần xuất hiện trung bình) phải không đổi. Ví dụ, nó có thể mô hình hóa số lượng khách hàng đến ngân hàng trong một giờ, số lượng email nhận được trong một ngày hoặc số lượng cuộc gọi điện thoại tại một trung tâm cuộc gọi mỗi phút.

### Định lý Bayes (Bayes' Theorem)
Định lý Bayes giúp cập nhật xác suất của một biến ngẫu nhiên dựa trên thông tin mới.
$$
P(A|B) = \frac{P(B|A)\cdot P(A)}{P(B)}
$$

* $P(A|B)$: Xác suất có điều kiện hoặc xác suất hậu nghiệm của biến ngẫu nhiên A khi biết B.
* $P(B|A)$: Xác suất có điều kiện của B khi biết A.
* $P(A)$: là xác suất tiên nghiệm.
* $P(B)$: Xác suất của biến ngẫu nhiên B.

## Calculus
### Đạo hàm và tích phân:
#### Đạo hàm:
Đạo hàm của một hàm số là một đại lượng mô tả sự biến thiên của hàm tại một điểm nào đó.

Kí hiệu của đạo hàm:
$\frac{df(x)}{dx}$ hoặc $\frac{\partial }{\partial x}$

#### Tích phân
Tích phân biểu diễn sự tích lũy các đại lượng và có thể được coi là diện tích dưới một đường cong. Trong machine learning, tích phân thường được dùng để tính toán các đại lượng như diện tích, thể tích và xác suất tích lũy.

## References
[^1]: Catherine Dee, "What are vectors and how do they people apply to machine learning", https://www.algolia.com/blog/ai/what-are-vectors-and-how-do-they-apply-to-machine-learning/

[^2]: Numb3rs 219: Dark Matter, https://pi.math.cornell.edu/~numb3rs/kostyuk/num219.htm

[^3]: Vũ Hữu Tiệp, Ôn tập Xác Suất cho Machine Learning, https://machinelearningcoban.com/2017/07/09/prob/

[^4]: https://fsppm.fulbright.edu.vn/cache/Topic-1.2-2023-11-09-18431673.pdf

[^5]: Adam Hayes, https://www.investopedia.com/terms/p/probabilitydistribution.asp#:~:text=A%20probability%20distribution%20is%20a,minimum%20and%20maximum%20possible%20values.