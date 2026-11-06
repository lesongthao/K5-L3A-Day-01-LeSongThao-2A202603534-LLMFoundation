# Side lab LLM API — Bài tập & Phản ánh

Đã triển khai đầy đủ 7 hàm trong `src/optional_llm_api/solution/solution.py`, vượt qua 19/19 unit test (`pytest tests/test_solution.py -v`). Thực nghiệm bên dưới chạy trên `gemini-3.5-flash` và `gemini-3.5-flash-lite` qua OpenAI-compatible API của Google AI Studio (`reasoning_effort="minimal"`).

---

## Bài tập 2.1 — Độ nhạy của Temperature

Gọi `call_openai("Hãy kể cho tôi một sự thật thú vị về Việt Nam.", temperature=...)` với 4 mức `0.0`, `0.5`, `1.0`, `1.5` (kết quả đầy đủ lưu ở `temperature_results.json`):

| `temperature` | Độ trễ (s) | Chủ đề & đặc điểm phản hồi thực tế |
| ---: | ---: | --- |
| `0.0` | 3.75 | Xuất khẩu hạt điều/tiêu đen số 1, cà phê số 2 và câu chuyện **Cà phê Trứng** (cụ Nguyễn Văn Giảng tại khách sạn Metropole năm 1940). Diễn đạt chuẩn mực, mạch lạc. |
| `0.5` | 3.63 | Vẫn chọn chủ đề **Cà phê Robusta & Cà phê Trứng**, cấu trúc gần giống mức `0.0` nhưng thêm phần "bonus" về **Cà phê muối (Huế)** và **Cà phê cốt dừa**. |
| `1.0` | 4.53 | Đổi hẳn sang góc nhìn mới: **cây điều có nguồn gốc từ Đông Bắc Brazil**, đến thập niên 1980 mới trồng phổ biến ở Việt Nam và Việt Nam giữ vị trí số 1 nhờ tự chế tạo máy bóc vỏ/phân loại giá rẻ. |
| `1.5` | 3.99 | Kết hợp nhiều ý hơn (xuất khẩu điều, cà phê trứng/muối/cốt dừa, thêm thống kê **84 gói mì ăn liền/người/năm**), nhưng bắt đầu xuất hiện từ bị nhiễu do sampling đuôi phân phối: viết nhầm *"Cà phê Robust"* và *"đánh bông lòng đỏ trứng gà với **mật khẩu**/đường"*. |

**Bạn nhận thấy quy luật gì qua bốn phản hồi?**
> Ở `temperature = 0.0` và `0.5`, mô hình chọn các token có xác suất cao nhất nên bám sát câu trả lời phổ biến nhất ("cà phê trứng"), văn phong ổn định và an toàn. Khi tăng lên `1.0`, phân phối xác suất phẳng hơn giúp mô hình rẽ sang một chủ đề ít gặp hơn nhưng rất thú vị (lịch sử cây điều và công nghệ máy bóc vỏ). Đến `1.5`, độ ngẫu nhiên cao làm câu trả lời ghép nhiều mảng thông tin nhưng bắt đầu chọn phải các token xác suất thấp gây lỗi dùng từ (*"mật khẩu/đường"* thay vì *"mật ong/đường"*).

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt `temperature` trong khoảng **0.0 – 0.2** (ưu tiên `0.0`). Chatbot hỗ trợ khách hàng cần trả lời nhất quán về chính sách, giá cả, quy trình đổi trả và tuyệt đối tránh tự suy diễn hay dùng từ sai như hiện tượng ở `temperature = 1.5`. Sự "sáng tạo" không cần thiết trong tác vụ tra cứu và giải đáp quy định.

---

## Bài tập 2.2 — Đánh đổi Chi phí

Kịch bản: `10.000` người dùng/ngày × `3` lần gọi × `350` token/lần = **10.500.000 token/ngày** (10,5M token/ngày).

**Ước tính xem `gpt-6.1-terra` đắt hơn `gpt-6-luna` bao nhiêu lần cho workload này** (giá 11/2026: terra \$1.00 / \$5.00, luna \$0.10 / \$0.50 mỗi 1M token input / output):
> Cả đơn giá input (\$1.00 vs \$0.10) lẫn đơn giá output (\$5.00 vs \$0.50) của `gpt-6.1-terra` đều đắt gấp **10 lần** `gpt-6-luna`. Do đó với bất kỳ tỷ lệ chia input/output nào trong 350 token mỗi lượt, tổng chi phí mỗi ngày của `gpt-6.1-terra` luôn đắt gấp **10 lần** `gpt-6-luna`:
> - Nếu giả sử mỗi lượt gồm ~250 token input + ~100 token output (tổng 7,5M input + 3,0M output mỗi ngày):
>   - `gpt-6-luna`: $7{,}5 \times \$0.10 + 3{,}0 \times \$0.50 = \$0.75 + \$1.50 = \$2.25\text{/ngày}$ ($\approx \$67.50\text{/tháng}$).
>   - `gpt-6.1-terra`: $7{,}5 \times \$1.00 + 3{,}0 \times \$5.00 = \$7.50 + \$15.00 = \$22.50\text{/ngày}$ ($\approx \$675\text{/tháng}$).
> - Nếu tính toàn bộ 10,5M token theo giá output (như hàm `compare_models`): `gpt-6-luna` tốn **\$5.25/ngày**, còn `gpt-6.1-terra` tốn **\$52.50/ngày** (gấp 10 lần).

**Mô tả một trường hợp mà chi phí cao hơn của `gpt-6.1-terra` là xứng đáng, và một trường hợp `gpt-6-luna` là lựa chọn tốt hơn:**
> - **Khi `gpt-6.1-terra` xứng đáng:** Trợ lý rà soát hợp đồng pháp lý, tư vấn tài chính hoặc chẩn đoán lỗi hệ thống phức tạp cần suy luận nhiều bước và tuân thủ nhiều ràng buộc cùng lúc — nơi một lỗi suy luận gây thiệt hại lớn hơn rất nhiều so với mức chênh \$20/ngày.
> - **Khi `gpt-6-luna` tốt hơn:** Các tác vụ khối lượng lớn, cấu trúc rõ ràng như phân loại SMS spam/ham (như ở bài lab chính), gán nhãn cảm xúc review, tóm tắt ngắn hoặc bộ định tuyến (router) phân loại ý định khách hàng trước khi chuyển tiếp.

---

## Bài tập 2.3 — Trải nghiệm Người dùng với Streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?**
> Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp với người dùng có câu trả lời dài (chatbot hội thoại, trợ lý viết code, giải thích tài liệu): dù tổng thời gian sinh toàn bộ câu trả lời không giảm, thời gian chờ đến chữ đầu tiên (*Time to First Token*) giảm từ vài giây xuống vài trăm ms, giúp người dùng đọc được ngay và thấy hệ thống đang phản hồi. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm hoặc cần đầu ra có cấu trúc trọn vẹn trước khi chạy bước tiếp theo — ví dụ phân loại spam/ham chỉ trả 1 từ, trích xuất JSON để gọi hàm/lưu cơ sở dữ liệu, hoặc chạy batch đánh giá hàng nghìn mẫu trên server.
