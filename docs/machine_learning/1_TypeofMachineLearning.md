# #1. Type of Machine Learning

## Supervised Learning <sup><font size="2">[Details](./4_SupervisedLearningAlgorithms.md)</font></sup>

Là loại thuật toán được sử dụng nhiều nhất. Đặc điểm của `supervised learning` là đưa ra các ví dụ về thuật toán học tập để rút kinh nghiệm.

Supervised Learning là một phương pháp học máy trong đó Model được huấn luyện trên một tập dữ liệu đã được gán nhãn. Dự đoán đầu ra (outcome) của một dữ liệu mới (new input) dựa trên các cặp (input, outcome) đã biết từ trước.

Supervised learning gồm 2 loại:

* Regression algorithms
    * Là bài toán mà đầu ra cần dự đoán là một giá trị liên tục. Ví dụ dự đoán giá nhà dựa trên các đặc trưng như diện tích, số phòng, vị trí... Model sẽ dự đoán một giá trị số.
* Classification algorithms
    * Là bài toán mà đầu ra cần dự đoán là một class thuộc một tập hữu hạn các class. Ví dự phân loại ảnh là chó hoặc mèo.

Supervised Learning là nền tảng của nhiều ứng dụng thực tế trong lĩnh vực như nhận dạng giọng nói, xử lý ngôn ngữ tự nhiên, và hệ thống gợi ý.

## Unsupervised Learning <sup><font size="2">[Details](./5_UnsupervisedLearningAlgorithms.md)</font></sup>

Trong thuật toán này, chúng ta không biết được outcome hay label mà chỉ có dữ liệu đầu vào. Thuật toán unsupervised learning sẽ dựa vào cấu trúc của dữ liệu để thực hiện một công việc nào đó, ví dụ như phân nhóm (clustering) hoặc giảm số chiều của dữ liệu (dimension reduction) để thuận tiện trong việc lưu trữ và tính toán.

Unsupervised learning bao gồm:

* Clustering (phân cụm)
    * Là một bài toán phân nhóm toàn bộ dữ liệu thành các nhóm nhỏ dựa trên sự liên quan giữa các dữ liệu trong mỗi nhóm
* Anomaly detection (phát hiện bất thường)
    * Được sử dụng để phát hiện các sự kiện bất thường
* Dimensionality reduction
    * Cho phép lấy một tập dữ liệu lớn và nén nó thành một tập dữ liệu nhỏ hơn nhiều trong khi mất ít thông tin nhất có thể.


## Semi-supervised Learning <sup><font size="2">[Details](./6_SemiSupervisedLearning.md)</font></sup>

Semi-supervised learning là một phương pháp nằm giữa Supervised learning và Unsupervised learning, kết hợp dữ liệu có nhãn (thường là ít) và dữ liệu không có nhãn trong quá trình huấn luyện Model học máy. Trong trường hợp thiếu dữ liệu gán nhãn đây sẽ là một phương pháp hữu ích.

Semi-supervised learning bao gồm một số kỹ thuật:

* Self-Training:
    * Model ban đầu được huấn luyện trên tập dữ liệu gán nhãn, sau đó sử dụng chính các dự đoán của mình trên dữ liệu chưa gán nhãn để tạo thêm dữ liệu gán nhãn.
* Co-Training:
    * Sử dụng hai hoặc nhiều Model khác nhau để huấn luyện lẫn nhau, mỗi Model sử dụng các phần khác nhau của dữ liệu và dự đoán Model này được sử dụng làm huấn luyện cho Model kìa.
* Generative Model:
    * Sử dụng các Model sinh dữ liệu như Generative Adversarial Networks (GANs) hoặc Variational Autoencoders (VAEs) để tạo ra dữ liệu mới từ dữ liệu chưa gán nhãn.

Semi-Supervised Learning là một cách tiếp cận hiệu quả để tận dụng dữ liệu chưa gán nhãn, giúp các Model học máy trở nên mạnh mẽ hơn trong các tình huống thực tế.

## Reinforcement Learning <sup><font size="2">[Details](./7_ReinforcementLearning.md)</font></sup>

Là một phương pháp học máy tập trung tạo việc tạo ra một tác nhân (agent) để đưa ra các quyết định bằng cách tương tác với môi trường, thông qua quá trình này, tác nhân sẽ học cách tối ưu hóa các hành động của mình để đạt được phần thưởng cao nhất theo thời gian.

Một số thuật toán trong Reinforcement learning được kể đến như:

* Q-Learning: 
    * Là một thuật toán học tăng cường không cần Model, được sử dụng để tìm ra chính sách tối ưu bằng cách học các hàm giá trị hành động (Q-Function).
* State-Action-Reward-State-Action (SARSA):
    * Là một thuật toán học tăng cường theo Model, giống với Q-Learning nhưng với sự khác biệt chính là việc cập nhật hàm Q dựa trên hành động thực sự được chọn ở trạng thái mới.
* Bên cạnh đó còn có: Markov Decision Processes (MDPs), Policy and Value Iteration, Deep Q-Networks (DQN), Policy  Gradient Methods.

Reinforcement Learning được ứng dụng rộng rãi trong nhiều lĩnh vực như chơi game, điều khiển robot, và hệ thống gợi ý, nơi các quyết định tối ưu được học từ kinh nghiệm thông qua tương tác liên tục với môi trường.

## Machine Unlearning <sup><font size="2">[Details](./8_MachineUnlearning.md)</font></sup>

Machine unlearning là quá trình mà trong đó một Model học máy loại bỏ hoặc giảm thiểu ảnh hưởng của các dữ liệu nhất định đã được sử dụng trong quá trình huấn luyện. Đây là một khái niệm quan trọng trong bối cảnh bảo mật dữ liệu và quyền riêng tư, đặc biệt khi có yêu cầu pháp lý hoặc từ người dùng về việc xóa dữ liệu cá nhân.

### Cách thực hiện machine unlearning
1. Huấn luyện lại Model từ đầu: Đây là cách tiếp cận đơn giản nhất nhưng tốn kém về mặt tài nguyên và thời gian.
2. Sử dụng các thuật toán chuyên dụng: Một số thuật toán cho phép cập nhật Model mà không cần huấn luyện lại toàn bộ, chỉ cần thay đổi nhỏ để loại bỏ ảnh hưởng của dữ liệu cụ thể.
3. Phân tích và điều chỉnh trọng số: Điều chỉnh trọng số của Model sao cho ảnh hưởng của dữ liệu cần loại bỏ được giảm thiểu.

## References

1. [Machine Learning Introduction](https://www.coursera.org/specializations/machine-learning-introduction)
2. [Semi supervised learning](https://viblo.asia/p/lam-gi-khi-mo-hinh-hoc-may-thieu-du-lieu-co-nhan-phan-2-semi-supervised-learning-vyDZOREkKwj)
3. [Machine Unlearning](https://www.youtube.com/watch?v=eiZuQmImxEE&t=345s)
