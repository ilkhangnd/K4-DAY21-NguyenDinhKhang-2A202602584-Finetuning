# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

> Điều làm em ngạc nhiên nhất là fine-tuning không chỉ nằm ở việc chọn model rồi bấm train. Phần loss mask và cách chat template biến dữ liệu thành token ảnh hưởng rất lớn đến kết quả. Em cũng nhận thấy prompt tối ưu cho base model đã cải thiện rõ rệt so với prompt đơn giản, nên fine-tune không tự động là lựa chọn tốt hơn prompting.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

> Em mất nhiều thời gian nhất ở NB4 vì phải train thêm ba cấu hình đối chứng thay vì chỉ train một adapter chính. Ban đầu em nghĩ train model chính và đánh giá mới là phần lâu nhất, nhưng thực tế việc tạo phép so sánh công bằng — cùng số step và cùng ngân sách tham số — mới tốn thời gian hơn. Em cũng không dự đoán `wrong_lr` sẽ mất nhiều thời gian inference nhưng lại cho kết quả target bằng 0.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

> Trước lab này, em nghĩ train loss thấp thì model chắc chắn tốt hơn. Sau khi chạy lab, em thấy `attn_only` có train loss thấp hơn `correct` nhưng target score lại thấp hơn một chút. Quan trọng hơn, bản fine-tune tăng target từ 0.765 lên 0.970 nhưng regression lại giảm từ 0.7911 xuống 0.5222, nên verdict cuối cùng là FAILED. Em hiểu rằng phải đánh giá cả target, regression, format và latency trước khi kết luận.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

> Em dùng AI assistant để đọc yêu cầu lab, giải thích ý nghĩa của NB1–NB5, kiểm tra output notebook và nhắc về `EVAL_LIMIT`. AI giúp em nhận ra lượt chạy đầu với `EVAL_LIMIT=8` chỉ là smoke test, không phải kết quả được nộp. Tuy nhiên, AI không thể thay thế việc em tự đọc Gatekeeper và kiểm tra artefact; nếu chỉ tin vào hướng dẫn mà không xem kết quả verify, em có thể hiểu nhầm kết quả chạy thử là kết quả cuối.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

> Bước đầu tiên em làm sẽ là xác định rõ tác vụ, tiêu chí đánh giá và tạo tập eval cố định trước khi train. Sau đó em sẽ kiểm tra chất lượng dữ liệu, chat template và loss mask. Em sẽ xây dựng baseline prompting đủ mạnh trước, vì kết quả lab cho thấy fine-tune có thể tăng điểm ở tác vụ chính nhưng vẫn làm giảm năng lực tổng quát của model.