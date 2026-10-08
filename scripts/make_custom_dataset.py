#!/usr/bin/env python3
"""Generate Custom Domain Dataset for Bonus B2 (Fintech & Digital Banking Customer Support Triage).

Task: Phân loại yêu cầu hỗ trợ khách hàng ngân hàng số & Fintech Việt Nam thành JSON có cấu trúc.
Schema:
  intent: khoa_the | khieu_nai_giao_dich | hoan_phi | loi_ung_dung | xac_thuc_sinh_trac
  urgency: cao | trung_binh | thap
  product: thẻ tín dụng | thẻ ghi nợ | ứng dụng mobile banking | chuyển tiền nhanh 247 | ví điện tử | vay tiêu dùng
  sentiment: tieu_cuc | trung_tinh | tich_cuc
"""
from __future__ import annotations

import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEED = 20261008

INTENTS = {
    "khoa_the": [
        "bị mất ví cần khoá thẻ gấp",
        "nghi ngờ bị lộ thông tin thẻ, đề nghị khoá chiều thanh toán online",
        "thẻ bị nuốt tại cây ATM và có giao dịch bất thường, khoá ngay giùm",
        "tạm khoá thẻ giúp tôi",
        "khoá thẻ để kiểm tra bảo mật",
    ],
    "khieu_nai_giao_dich": [
        "tài khoản bị trừ tiền 2 lần cho cùng một giao dịch",
        "giao dịch báo thành công nhưng bên nhận chưa thấy tiền về",
        "chuyển nhầm tiền qua số tài khoản khác nhờ ngân hàng tra soát",
        "xuất hiện giao dịch lạ tại nước ngoài dù tôi đang ở nhà",
        "quẹt thẻ POS bị lỗi nhưng tin nhắn báo trừ tiền",
    ],
    "hoan_phi": [
        "đề nghị hoàn lại phí duy trì tài khoản tháng này",
        "khiếu nại bị trừ phí thường niên không đúng thông báo",
        "xin miễn giảm phí chuyển tiền phát sinh ngoài ý muốn",
        "yêu cầu hoàn phí SMS banking vì tôi đã hủy dịch vụ",
        "hoàn tiền phí phát hành thẻ vật lý",
    ],
    "loi_ung_dung": [
        "app bị văng ra liên tục sau khi cập nhật phiên bản mới",
        "không thể đăng nhập do hệ thống báo lỗi máy chủ",
        "chức năng quét mã QR thanh toán không nhận diện được",
        "màn hình bị đơ khi bấm xác nhận chuyển khoản",
        "không nhận được mã thông báo Smart OTP",
    ],
    "xac_thuc_sinh_trac": [
        "không quét được chip NFC thẻ căn cước CCCD",
        "nhận diện khuôn mặt liên tục báo không khớp với dữ liệu",
        "đổi điện thoại mới nên chưa kích hoạt lại được sinh trắc học",
        "hướng dẫn cài đặt vân tay để chuyển tiền trên 10 triệu",
        "lỗi xác thực sinh trắc học khi thực hiện chuyển tiền hạn mức lớn",
    ],
}

URGENCY_MARKERS = {
    "cao": ["gấp khẩn cấp", "xử lý ngay lập tức", "rất khẩn trương", "ngay bây giờ", "trong ngày hôm nay"],
    "trung_binh": ["sớm giúp tôi", "mong phản hồi sớm", "trong tuần này", "khi có ca trực"],
    "thap": ["khi nào rảnh thì xem", "không vội", "tham khảo thông tin thôi", "tiện thể kiểm tra giúp"],
}

SENTIMENTS = {
    "tieu_cuc": ["rất bức xúc", "dịch vụ quá thất vọng", "ảnh hưởng nghiêm trọng tới công việc", "bực mình với hệ thống"],
    "trung_tinh": ["nhờ ngân hàng kiểm tra", "hỗ trợ tra soát giúp mình", "xin hướng dẫn chi tiết", "cho tôi hỏi thông tin"],
    "tich_cuc": ["cảm ơn ngân hàng nhiều", "nhân viên tư vấn nhiệt tình", "vẫn luôn tin dùng dịch vụ", "mong được hỗ trợ tốt như mọi khi"],
}

PRODUCTS = [
    "thẻ tín dụng",
    "thẻ ghi nợ",
    "ứng dụng mobile banking",
    "chuyển tiền nhanh 247",
    "ví điện tử",
    "vay tiêu dùng",
]

OPENERS = ["Kính gửi ngân hàng,", "Chào CSKH,", "Alo tổng đài,", "Em ơi hỗ trợ anh,", "Chào admin,"]

INSTRUCTION = (
    "Phân loại yêu cầu hỗ trợ khách hàng ngân hàng số sau thành JSON với đúng 4 khóa: "
    "intent, urgency, product, sentiment. Chỉ trả về JSON, không giải thích.\n\n"
    "intent thuộc: khoa_the | khieu_nai_giao_dich | hoan_phi | loi_ung_dung | xac_thuc_sinh_trac\n"
    "urgency thuộc: cao | trung_binh | thap\n"
    "sentiment thuộc: tieu_cuc | trung_tinh | tich_cuc\n"
    "product: tên sản phẩm dịch vụ xuất hiện trong yêu cầu."
)


def make_custom_ticket(rng: random.Random) -> dict:
    intent = rng.choice(list(INTENTS))
    urgency = rng.choice(list(URGENCY_MARKERS))
    sentiment = rng.choice(list(SENTIMENTS))
    product = rng.choice(PRODUCTS)
    txn_id = f"TXN{rng.randint(10000000, 99999999)}"

    body = (
        f"{rng.choice(OPENERS)} tôi đang sử dụng dịch vụ {product} mã giao dịch {txn_id}. "
        f"{rng.choice(INTENTS[intent]).capitalize()}. "
        f"{rng.choice(URGENCY_MARKERS[urgency]).capitalize()}. "
        f"{rng.choice(SENTIMENTS[sentiment]).capitalize()}."
    )
    label = {"intent": intent, "urgency": urgency, "product": product, "sentiment": sentiment}
    return {
        "instruction": INSTRUCTION,
        "input": body,
        "output": json.dumps(label, ensure_ascii=False),
        "label": label,
        "txn_id": txn_id,
    }


def main():
    rng = random.Random(SEED)
    records = []
    # Generate 250 samples
    for _ in range(250):
        records.append(make_custom_ticket(rng))

    out_file = DATA / "custom_dataset.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} custom domain samples to {out_file}")


if __name__ == "__main__":
    main()
