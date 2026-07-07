# News Search Engine — Hệ thống tìm kiếm cho trang báo chí số
### Tài liệu thiết kế & Checklist nghiệm thu

> *Tài liệu tư vấn kiến trúc & vận hành. Các con số chi phí là ước tính tham khảo tại tháng 6/2026, cần kiểm chứng lại theo báo giá thực tế của nhà cung cấp và tỷ giá tại thời điểm triển khai.*

## Mục lục
1. [Hệ thống làm gì?](#1-hệ-thống-làm-gì)
2. [Danh sách chức năng chính](#2-danh-sách-chức-năng-chính)
3. [Ví dụ sử dụng thực tế](#3-ví-dụ-sử-dụng-thực-tế)
4. [Công nghệ sử dụng](#4-công-nghệ-sử-dụng)
5. [Bảng giá tham khảo](#5-bảng-giá-tham-khảo)
6. [Ước tính chi phí xử lý](#6-ước-tính-chi-phí-xử-lý)
7. [Checklist nghiệm thu](#7-checklist-nghiệm-thu)
8. [Điều kiện vận hành](#8-điều-kiện-vận-hành)

---

## 1. Hệ thống làm gì?

**News Search Engine** giúp biến kho bài viết báo chí thành dữ liệu có thể tìm kiếm thông minh. Thay vì đọc/lọc thủ công, người dùng có thể:

- Gõ **từ khóa ngắn** hoặc **câu hỏi tiếng Việt** → hệ thống trả về danh sách bài viết liên quan, xếp theo mức độ phù hợp
- **Lọc theo metadata** → tác giả, chuyên mục, khoảng thời gian, nguồn
- **Ưu tiên tin mới** cho truy vấn thời sự (cùng truy vấn, kết quả đổi theo thời điểm)
- **Gom bài trùng/gần trùng** về cùng một sự kiện, đảm bảo kết quả đa dạng góc nhìn

Hệ thống hoạt động qua **2 giai đoạn**:

| Giai đoạn | Mô tả đơn giản |
|---|---|
| **Index (xử lý bài viết)** | Hệ thống "đọc" bài một lần: làm sạch nội dung, trích xuất thực thể, sinh embedding, phát hiện trùng lặp, lưu vào chỉ mục |
| **Search (tìm kiếm)** | Người dùng truy vấn → hệ thống trả về danh sách bài viết phù hợp (JSON thống nhất) |

---

## 2. Danh sách chức năng chính

### 2.1 Khởi tạo hệ thống

| # | Chức năng | Mô tả |
|---|---|---|
| **F-01** | Khởi tạo cơ sở dữ liệu | Tạo sẵn chỉ mục từ khóa (BM25), kho vector và đồ thị tri thức trước khi index bài viết |

### 2.2 Index — Xử lý & lưu trữ bài viết

| # | Chức năng | Mô tả | Đầu ra |
|---|---|---|---|
| **F-02** | Thu nhận & chuẩn hóa bài viết | Nhận bài từ CMS/nguồn, làm sạch HTML, chuẩn hóa metadata (tiêu đề, tác giả, chuyên mục, thời gian đăng) | Bài viết đã chuẩn hóa |
| **F-03** | Tách & token hóa nội dung | Tách tiêu đề/sapo/nội dung, tokenization tiếng Việt (xử lý dấu, tách từ) | Trường văn bản đã token hóa |
| **F-04** | Đánh chỉ mục từ khóa | Xây inverted index BM25 phục vụ khớp chính xác & tên riêng | Chỉ mục lexical |
| **F-05** | Sinh embedding ngữ nghĩa | Vector hóa nội dung bài viết phục vụ tìm theo ngữ nghĩa | Vector lưu trong kho vector |
| **F-06** | Trích xuất thực thể (NER) | Nhận diện tên người, tổ chức, địa điểm, sự kiện trong bài | Knowledge graph (đồ thị tri thức) |
| **F-07** | Phát hiện trùng/gần trùng | Đối chiếu bài mới với kho hiện có, gom các bài cùng sự kiện thành cụm | Nhãn cụm sự kiện & cờ trùng lặp |
| **F-08** | Cập nhật / gỡ bài | Đồng bộ khi bài được sửa hoặc gỡ (unpublish) — cập nhật/xóa khỏi mọi chỉ mục | Chỉ mục đồng bộ thời gian thực |

### 2.3 Search — Tìm kiếm

> **Định dạng đầu ra thống nhất:** Mọi loại tìm kiếm đều trả về **danh sách JSON** — mỗi phần tử là một bài viết với đúng **5 trường**:
> `article_id` , `title` , `url` , `published_at` , `snippet`
>
> *(Điểm xếp hạng, embedding, thực thể… được dùng nội bộ để sắp xếp nhưng không xuất ra response.)*

| # | Chức năng | Mô tả | Đầu vào |
|---|---|---|---|
| **F-09** | Tìm kiếm bằng từ khóa | Khớp chính xác, tên riêng, thực thể hiếm (BM25) | Từ khóa ngắn |
| **F-10** | Tìm kiếm bằng câu hỏi tự nhiên | Hiểu đồng nghĩa, khái niệm, câu hỏi dài (semantic/vector) | Câu hỏi tiếng Việt |
| **F-11** | Tìm kiếm kết hợp (hybrid) | Trộn kết quả từ khóa + ngữ nghĩa bằng RRF, rerank tinh top đầu | Truy vấn bất kỳ |
| **F-12** | Lọc theo metadata | Giới hạn theo tác giả / chuyên mục / khoảng thời gian / nguồn | Bộ lọc |
| **F-13** | Ưu tiên độ mới (time-decay) | Tăng điểm bài mới cho truy vấn thời sự (Query Deserves Freshness) | — |
| **F-14** | Gom cụm & đa dạng hóa | Bỏ bài gần trùng, đảm bảo top-k đa góc nhìn (dedup + MMR) | — |
| **F-15** | Trích đoạn nổi bật (snippet) | Sinh `snippet` bôi đậm đoạn khớp truy vấn | — |

---

## 3. Ví dụ sử dụng thực tế

### Ví dụ 1 — Tìm kiếm bằng từ khóa
**Tình huống:** Biên tập viên tìm bài về số liệu lạm phát mới công bố.
**Thao tác:** Nhập `"lạm phát tháng 6"`
**Kết quả trả về (JSON):**
```json
[
  {
    "article_id": "eco-8821",
    "title": "CPI tháng 6 tăng 3,2%, lạm phát trong tầm kiểm soát",
    "url": "/kinh-te/cpi-thang-6-lam-phat-8821.html",
    "published_at": "2026-07-06T08:15:00+07:00",
    "snippet": "...chỉ số giá tiêu dùng (CPI) tháng 6 tăng 3,2% so với cùng kỳ, <b>lạm phát</b> cơ bản ở mức..."
  },
  {
    "article_id": "eco-8790",
    "title": "Chuyên gia: áp lực lạm phát nửa cuối năm đến từ giá năng lượng",
    "url": "/kinh-te/ap-luc-lam-phat-8790.html",
    "published_at": "2026-07-05T19:40:00+07:00",
    "snippet": "...áp lực <b>lạm phát</b> nửa cuối năm chủ yếu đến từ biến động giá xăng dầu..."
  }
]
```
**Giá trị:** Trả đúng bài mới nhất về đúng thực thể/số liệu, không lẫn bài cũ.

### Ví dụ 2 — Tìm kiếm bằng câu hỏi tự nhiên
**Tình huống:** Truy vấn dài, ít khớp từ khóa đúng nghĩa.
**Thao tác:** Nhập `"vì sao giá vàng tăng mạnh tuần này"`
**Kết quả trả về (JSON):**
```json
[
  {
    "article_id": "eco-8905",
    "title": "Giá vàng lập đỉnh: ba nguyên nhân chính đẩy giá đi lên",
    "url": "/kinh-te/gia-vang-lap-dinh-8905.html",
    "published_at": "2026-07-07T09:05:00+07:00",
    "snippet": "...vàng tăng do kỳ vọng hạ lãi suất, lực mua trú ẩn và đồng USD suy yếu..."
  }
]
```
**Giá trị:** Tầng ngữ nghĩa tìm được bài *giải thích nguyên nhân* dù không chứa đúng cụm từ khóa; hybrid + rerank đẩy bài phân tích lên đầu.

### Ví dụ 3 — Truy vấn thời sự (đổi ý định theo thời gian)
**Tình huống:** Vừa xảy ra sự kiện trong ngày.
**Thao tác:** Nhập `"động đất"`
**Kết quả trả về (JSON):** *(intent = tin nóng → time-decay ưu tiên bài < 24h)*
```json
[
  {
    "article_id": "news-9910",
    "title": "Động đất 5,1 độ tại Kon Tum, người dân cảm nhận rung lắc",
    "url": "/thoi-su/dong-dat-kon-tum-9910.html",
    "published_at": "2026-07-07T06:22:00+07:00",
    "snippet": "...trận <b>động đất</b> mạnh 5,1 độ xảy ra sáng nay tại khu vực..."
  }
]
```
**Giá trị:** Cùng truy vấn `"động đất"` — nếu không có sự kiện gần, hệ thống trả bài nền/giải thích chung; khi có tin nóng, ưu tiên bài mới nhất.

### Ví dụ 4 — Tìm kiếm kết hợp bộ lọc metadata
**Tình huống:** Chỉ tìm trong chuyên mục Kinh tế, từ đầu tháng.
**Thao tác:**
```
Nhập từ khóa: "giá xăng"
Lọc: category = "Kinh tế", from = "2026-07-01"
```
**Kết quả trả về (JSON):** chỉ gồm bài thuộc chuyên mục Kinh tế đăng từ 01/07, cùng định dạng 5 trường.

### Ví dụ 5 — Sự kiện nhiều bài trùng (gom cụm & đa dạng hóa)
**Tình huống:** Sự kiện lớn, hàng trăm bài gần trùng.
**Thao tác:** Nhập `"kết quả bầu cử"`
**Kết quả trả về (JSON):** hệ thống gom bài trùng về từng cụm, top-10 gồm: bản tin kết quả chính, bài phân tích, phản ứng quốc tế… thay vì 10 bản tin giống nhau (dedup + MMR hoạt động).

---

## 4. Công nghệ sử dụng

### 4.1 Tóm tắt cho khách hàng

| Nhóm | Công nghệ | Vai trò | Chi phí |
|---|---|---|---|
| Tìm kiếm từ khóa | OpenSearch (BM25/Lucene) | Khớp chính xác, tên riêng, lọc metadata | Hạ tầng server (tự host hoặc cloud) |
| AI ngữ nghĩa | OpenAI text-embedding-3-large | Tìm theo ý nghĩa, câu hỏi tự nhiên | Trả theo lượt sử dụng (API) |
| AI hiểu truy vấn | OpenAI (GPT) | Sửa lỗi, viết lại/mở rộng truy vấn, trích xuất thực thể | Trả theo lượt sử dụng (API) |
| Xếp hạng tinh (rerank) | BGE-reranker-v2-m3 | Sắp xếp lại top kết quả chính xác cao | Miễn phí (chạy trên server) |
| Cơ sở dữ liệu vector | Milvus | Lưu & tìm nhanh theo ngữ nghĩa | Hạ tầng server (tự host hoặc cloud) |
| Đồ thị tri thức | Neo4j | Liên kết người – tổ chức – sự kiện – bài viết | Hạ tầng server (tự host hoặc cloud) |
| Phát hiện trùng lặp | MinHash-LSH / SimHash | Gom bài gần trùng cùng sự kiện | Miễn phí (chạy trên server) |
| Xử lý tiếng Việt | underthesea | Tách từ, NER offline (tùy chọn thay GPT) | Miễn phí (chạy trên server) |

### 4.2 Chi tiết model AI sử dụng

| Chức năng | Model | Nhà cung cấp | Chạy ở đâu |
|---|---|---|---|
| Sinh embedding bài viết & truy vấn | text-embedding-3-large | OpenAI | Cloud (API) |
| Viết lại / mở rộng truy vấn | gpt-4.1-mini | OpenAI | Cloud (API) |
| Trích xuất thực thể (NER) | gpt-4.1-mini *(hoặc underthesea local)* | OpenAI / VN-NLP | Cloud (API) / Server |
| Xếp hạng tinh (rerank) | BGE-reranker-v2-m3 | BAAI | Server (local) |
| Tìm kiếm từ khóa | BM25 (Lucene) | OpenSearch | Server (local) |
| Phát hiện trùng lặp | MinHash-LSH | — | Server (local) |

---

## 5. Bảng giá tham khảo

> **Lưu ý:** Giá API OpenAI có thể thay đổi. Số liệu dưới đây tham khảo tại **tháng 6/2026**. Giá hạ tầng (OpenSearch, Milvus, Neo4j, server) phụ thuộc quy mô triển khai.

### 5.1 OpenAI API — Giá theo 1 triệu token

| Model | Input | Output | Dùng cho |
|---|---|---|---|
| gpt-4.1-mini | $0.40 | $1.60 | Viết lại truy vấn, trích xuất thực thể |
| text-embedding-3-large | $0.13 | — | Embedding bài viết & truy vấn |

### 5.2 Công nghệ miễn phí (chạy trên server)

| Công nghệ | Chi phí license | Chi phí vận hành |
|---|---|---|
| OpenSearch (self-hosted) | Miễn phí (open source) | RAM + disk server |
| BGE-reranker-v2-m3 | Miễn phí | CPU/GPU server |
| Milvus (self-hosted) | Miễn phí (open source) | RAM + disk server |
| Neo4j (Community) | Miễn phí | RAM + disk server |
| underthesea, MinHash-LSH | Miễn phí | CPU server |

---

## 6. Ước tính chi phí xử lý

### 6.1 Giả định
- Bài viết trung bình **~800 từ (~1.500 token)**
- Kho bài: **~1.000.000 bài**, cập nhật **~1.500 bài/ngày**
- Lưu lượng: **~100.000 truy vấn/ngày**

### 6.2 Chi phí API khi index 1 bài viết

| Bước xử lý | Model | Số lượt gọi | Chi phí ước tính |
|---|---|---|---|
| Sinh embedding bài viết | text-embedding-3-large | 1 bài (~1.500 token) | ~$0.0002 |
| Trích xuất thực thể (NER) | gpt-4.1-mini | 1 bài | ~$0.0008 – $0.0015 |
| Đánh chỉ mục BM25 | OpenSearch | 1 bài | $0 (local) |
| Phát hiện trùng lặp | MinHash-LSH | 1 bài | $0 (local) |
| **Tổng ước tính** | | | **~$0.001 – $0.002 / bài** (có NER) — hoặc **~$0.0002 / bài** nếu NER chạy local (underthesea) |

### 6.3 Chi phí tìm kiếm (mỗi lần query)

| Loại tìm kiếm | Chi phí ước tính |
|---|---|
| Tìm bằng từ khóa (BM25) | $0 (chạy local) |
| Tìm bằng câu hỏi tự nhiên | ~$0.000004 (1 embedding truy vấn) |
| Có viết lại truy vấn (LLM) | ~$0.0001 – $0.0003 |
| Rerank top kết quả | $0 (chạy local) |

**Kết luận:** Chi phí chủ yếu nằm ở **giai đoạn index** (xử lý một lần). Giai đoạn **tìm kiếm gần như miễn phí** (trừ embedding truy vấn và LLM rewrite tùy chọn). Có thể **cache** kết quả truy vấn phổ biến (tin nóng) để giảm thêm chi phí.

### 6.4 Bảng quy mô tham khảo

| Quy mô kho | Chi phí index API (chỉ embedding) | Chi phí index API (embedding + NER GPT) |
|---|---|---|
| 100.000 bài | ~$20 | ~$100 – $200 |
| 500.000 bài | ~$100 | ~$500 – $1.000 |
| 1.000.000 bài | ~$200 | ~$1.000 – $2.000 |

*Chi phí cập nhật hằng ngày (~1.500 bài/ngày): ~$0.3/ngày (embedding) hoặc ~$1.5 – $3/ngày (kèm NER GPT) → ~$10 – $90/tháng. Chưa bao gồm chi phí hạ tầng server.*

---

## 7. Checklist nghiệm thu

### 7.1 Khởi tạo hệ thống
- [ ] **F-01** Khởi tạo thành công chỉ mục BM25 (OpenSearch), kho vector (Milvus) và đồ thị tri thức (Neo4j)
- [ ] Kết nối OpenAI API hoạt động (API key hợp lệ)
- [ ] Kết nối OpenSearch / Milvus hoạt động
- [ ] Kết nối Neo4j hoạt động (hoặc tắt được qua cấu hình)

### 7.2 Index — Xử lý bài viết
- [ ] **F-02** Bài viết được thu nhận & chuẩn hóa metadata (tiêu đề, tác giả, chuyên mục, thời gian đăng)
- [ ] **F-03** Nội dung được tách & token hóa tiếng Việt đúng (xử lý dấu, tên riêng)
- [ ] **F-04** Bài được đánh chỉ mục BM25, tìm được bằng từ khóa
- [ ] **F-05** Embedding bài viết được sinh và lưu vào kho vector
- [ ] **F-06** Thực thể (người, tổ chức, địa điểm, sự kiện) được trích xuất và lưu vào đồ thị tri thức
- [ ] **F-07** Bài trùng/gần trùng được phát hiện và gom cụm sự kiện
- [ ] **F-08** Bài **cập nhật** được đồng bộ; bài **gỡ (unpublish)** biến mất khỏi mọi chỉ mục
- [ ] Index lại bài (re-index) không tạo dữ liệu trùng lặp

**Tiêu chí đạt:**
- Index lô ≥ 100 bài mẫu hoàn tất không lỗi
- Bài mới xuất bản tìm được trong ≤ **X phút** (thống nhất SLA, ví dụ 2 phút)
- Bài đã gỡ không còn xuất hiện trong kết quả tìm kiếm
- Cụm bài cùng sự kiện được gom đúng (kiểm tra thủ công ≥ 5 sự kiện)

### 7.3 Search — Tìm kiếm

**Tiêu chí đạt (test với kho đã index):**

| Test case | Input | Kỳ vọng |
|---|---|---|
| Keyword search | `"lạm phát tháng 6"` | Trả về ≥ 1 bài liên quan; JSON đúng 5 trường |
| Natural query | `"vì sao giá vàng tăng"` | Trả bài giải thích liên quan dù không khớp từ khóa; đúng 5 trường |
| Freshness (tin nóng) | `"động đất"` (có sự kiện trong ngày) | Bài < 24h được ưu tiên lên đầu |
| Filter metadata | Text + `category="Kinh tế"` + `from` | Chỉ trả bài đúng chuyên mục & khoảng thời gian |
| Dedup & đa dạng | `"kết quả bầu cử"` | Không có ≥ 2 bài gần trùng trong top-10 |
| Consistency | Hai truy vấn gần giống nhau | Kết quả nhất quán, không lệch bất thường |
| Empty input | Không nhập gì | Báo lỗi rõ ràng |
| Output schema | Bất kỳ loại search nào | Chỉ 5 trường quy định (không lộ score/embedding/entity) |

- [ ] **F-09 → F-15** hoạt động đúng theo bảng test case trên
- [ ] **nDCG@10 ≥ ngưỡng thống nhất** trên bộ đánh giá (ví dụ ≥ 0.6, chốt theo baseline)

### 7.4 Hiệu năng & ổn định
- [ ] Tìm kiếm trả kết quả trong **< 300ms (p95)**, chấp nhận < 500ms (p99)
- [ ] Chịu tải đỉnh (ví dụ 20 QPS) không lỗi, không tăng latency quá ngưỡng
- [ ] Độ trễ đánh chỉ mục bài mới đạt SLA
- [ ] Hệ thống xử lý lỗi gracefully (bài thiếu nội dung, truy vấn rỗng, ký tự lạ…)
- [ ] Log ghi nhận đủ thông tin để debug + log click/vị trí phục vụ đánh giá

---

## 8. Điều kiện vận hành

### 8.1 Yêu cầu kỹ thuật

| Hạng mục | Yêu cầu tối thiểu |
|---|---|
| Python | 3.10+ |
| RAM | 32 GB (đủ giữ inverted index + vector nóng; nên quantize vector) |
| GPU | Không bắt buộc (tăng tốc embedding local & reranker) |
| Disk | Tùy quy mô kho bài (SSD khuyến nghị) |
| OpenAI API Key | Bắt buộc (nếu dùng embedding/NER qua API) |
| OpenSearch | Bắt buộc (có replica + snapshot) |
| Milvus | Bắt buộc (self-hosted hoặc Zilliz Cloud) |
| Neo4j | Bắt buộc (hoặc tắt được qua cấu hình nếu chưa dùng knowledge graph) |
| Nguồn dữ liệu | Kết nối CMS + webhook/queue khi bài xuất bản/sửa/gỡ |

### 8.2 Định dạng dữ liệu hỗ trợ

| Định dạng | Hỗ trợ |
|---|---|
| Bài viết HTML (từ CMS) | ✅ (làm sạch tự động) |
| JSON / API ingest | ✅ |
| Metadata bắt buộc | `title`, `body`, `author`, `category`, `published_at`, `url`, `status` |
| Bài không đủ metadata | ✅ index được nhưng giảm khả năng lọc/xếp hạng |

### 8.3 Ngôn ngữ

| Hạng mục | Ngôn ngữ |
|---|---|
| Token hóa & tìm kiếm nội dung | Tiếng Việt (xử lý dấu, tách từ, alias, tên riêng) |
| Trích xuất thực thể (NER) | Tiếng Việt + tiếng Anh (tên riêng quốc tế) |
| Câu hỏi tự nhiên | Tiếng Việt |
| Embedding ngữ nghĩa | Đa ngữ (Việt/Anh) |

---

*Ba điểm đặc thù báo chí đã đưa vào thiết kế (khác với search thông thường): **time-decay/QDF** cho tin nóng (F-13), **dedup gom cụm sự kiện** (F-07, F-14), và **đồng bộ gỡ bài thời gian thực** (F-08). Nếu chưa cần đồ thị tri thức ngay, Neo4j và NER có thể tắt để giảm chi phí, thêm sau.*
