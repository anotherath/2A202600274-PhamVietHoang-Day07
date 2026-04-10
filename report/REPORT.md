# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Việt Hoàng
**Nhóm:** Vinfast c2
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector có cosine similarity cao nghĩa là chúng trỏ về cùng một hướng trong không gian vector, biểu thị hai văn bản có ý nghĩa tương đồng hoặc liên quan chặt chẽ về mặt ngữ nghĩa.

**Ví dụ HIGH similarity:**
- Sentence A: "Python là ngôn ngữ lập trình phổ biến cho AI"
- Sentence B: "Ngôn ngữ Python được sử dụng rộng rãi trong machine learning"
- Tại sao tương đồng: Cả hai câu đều nói về Python và ứng dụng trong AI/machine learning

**Ví dụ LOW similarity:**
- Sentence A: "Chính sách bán hàng của VinFast tháng 3/2026"
- Sentence B: "Python là ngôn ngữ lập trình dễ học"
- Tại sao khác: Hai câu nói về chủ đề hoàn toàn khác nhau (bán hàng ô tô vs lập trình)

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ quan tâm đến hướng của vector (góc giữa hai vector) mà không phụ thuộc vào độ lớn, giúp so sánh tương đối giữa các văn bản dù độ dài khác nhau. Euclidean distance bị ảnh hưởng bởi độ lớn của vector, dễ bị sai lệch khi so sánh văn bản ngắn và dài.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> - `num_chunks = ceil((10000 - 50) / (500 - 50))`
> - `num_chunks = ceil(9950 / 450)`
> - `num_chunks = ceil(22.11) = 23`
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số chunk sẽ tăng lên: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25` chunks.
> 
> Overlap nhiều hơn giúp bảo toàn ngữ cảnh ở ranh giới giữa các chunk, tránh mất thông tin quan trọng khi câu bị cắt đột ngột. Đặc biệt hữu ích với các tài liệu kỹ thuật có các bước thực hiện liên tiếp hoặc định nghĩa trải dài qua nhiều câu.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Chính sách bán hàng và ưu đãi Ô tô điện VinFast tại Việt Nam

**Tại sao nhóm chọn domain này?**
> Domain này được chọn vì có nguồn tài liệu chính thức phong phú, cấu trúc rõ ràng với nhiều bảng giá và điều kiện áp dụng cụ thể. Đây là bài toán RAG thực tế mà các đại lý ô tô cần: trả lờI nhanh các câu hỏi về giá, khuyến mãi, điều kiện áp dụng chính sách cho khách hàng.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | tailieu.md | Chính sách VinFast | ~80,000 | type: policy, language: vi, domain: automotive |
| 2 | python_intro.txt | Python documentation | ~1,500 | type: technical, language: en, domain: programming |
| 3 | vector_store_notes.md | RAG documentation | ~1,800 | type: technical, language: en, domain: ai |
| 4 | rag_system_design.md | RAG architecture | ~1,600 | type: technical, language: en, domain: ai |
| 5 | customer_support_playbook.txt | Support guidelines | ~1,200 | type: guide, language: en, domain: support |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "data/tailieu.md" | Xác định nguồn gốc tài liệu để truy vết thông tin |
| extension | string | ".md" | Lọc theo định dạng file khi cần |
| type | string | "policy" | Phân loại tài liệu (chính sách, kỹ thuật, hướng dẫn) |
| language | string | "vi" | Lọc theo ngôn ngữ cho câu hỏi tiếng Việt/Anh |
| domain | string | "automotive" | Phân biệt lĩnh vực để tránh nhầm lẫn |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| tailieu.md | FixedSizeChunker (`fixed_size`) | 45 | 485 | Không - hay cắt giữa câu |
| tailieu.md | SentenceChunker (`by_sentences`) | 38 | 520 | Tốt - theo ranh giới câu |
| tailieu.md | RecursiveChunker (`recursive`) | 42 | 495 | Rất tốt - giữ cấu trúc đoạn |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> RecursiveChunker thử tách văn bản theo thứ tự ưu tiên các separator: `\n\n` (đoạn văn), `\n` (dòng mới), `. ` (câu), ` ` (từ), và cuối cùng là tách cứng theo kích thước. Nếu phần tách được còn lớn hơn chunk_size, nó đệ quy xuống separator nhỏ hơn. Cách này giữ được cấu trúc phân cấp của văn bản markdown với các section và subsection.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu chính sách VinFast có cấu trúc rõ ràng với các section (###), bullet points, và bảng. RecursiveChunker ưu tiên tách theo đoạn văn và section giúp giữ nguyên ý nghĩa của từng chính sách mà không bị cắt giữa chừng, đồng thời vẫn đảm bảo chunk không quá dài.

**Code snippet:**
```python
# Sử dụng RecursiveChunker mặc định từ src/chunking.py
chunker = RecursiveChunker(chunk_size=500)
chunks = chunker.chunk(document_content)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| tailieu.md | SentenceChunker (best baseline) | 38 | 520 | Tốt |
| tailieu.md | **RecursiveChunker (của tôi)** | 42 | 495 | **Rất tốt** - giữ cấu trúc markdown |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Phạm Việt Hoàng | RecursiveChunker (openai api) | 8 | Đưa ra khá đúng ngữ nghĩa trong đa số trường hợp | Chunk có độ dài ổn định, đủ content để LLM hiểu |
| Nguyễn Thùy Linh | Parent-Child / Small-to-Big | 4 | Bảo toàn ngữ cảnh toàn policy; cân bằng fact retrieval + context; phù hợp với câu hỏi multi-condition VinFast | Cần metadata filtering để tối ưu; mock embedder làm giảm chất lượng retrieval |
| Phan Tuấn Minh | CustomChunker + Local Embed | 6 | Giữ nguyên bảng giá, không cắt ngang | Chunk lớn (755 ký tự), embedding match nhầm keyword |
| Bùi Minh Ngọc | SentenceChunker (3 câu/chunk) | 6 | Giữ nguyên ý nghĩa từng điều khoản, không cắt đứt câu | Chunk có thể rất dài nếu câu văn dài; bảng Markdown bị xử lý kém |
| Phạm Đình Trường | RecursiveChunker (Separator-based) | 3 | Bảo toàn ngữ cảnh theo cấu trúc tự nhiên (đoạn văn, câu); giữ được tính logic của các điều khoản chính sách VinFast; tránh cắt vụn thông tin quan trọng. | Mock embedder khiến kết quả truy xuất thực tế bị sai lệch; cần tinh chỉnh danh sách Separators để đạt hiệu quả cao nhất với tiếng Việt. |
| Việt Anh | SentenceChunker (By sentences) | 3 | Phân mảnh dựa trên đơn vị câu giúp giữ trọn vẹn ý nghĩa của từng phát biểu; cấu trúc chunk gọn gàng, dễ đọc cho AI. | Dễ làm mất ngữ cảnh liên kết giữa các câu nếu chúng bị chia vào các chunk khác nhau; hiệu quả tìm kiếm thấp do Embedder không hiểu ngữ nghĩa. |
| Lê Đức Thanh | SentenceChunker | 0/5 | Giữ ý theo câu, dễ đọc, dễ giải thích | Có thể chưa tối ưu nếu câu quá dài hoặc nhiều bảng dữ liệu |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker là tốt nhất cho domain chính sách VinFast vì tài liệu có cấu trúc phân cấp rõ ràng với các section và subsection. Strategy này tôn trọng ranh giới tự nhiên của tài liệu trong khi vẫn kiểm soát được kích thước chunk.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex `(?<=[.!?])\s+` để tách câu dựa trên dấu câu (. ! ?) và khoảng trắng sau đó. Lookbehind giữ lại dấu câu trong câu được tách. Sau đó nhóm các câu thành chunk theo `max_sentences_per_chunk`. Edge case xử lý: loại bỏ câu rỗng sau khi strip whitespace.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy: nếu text ngắn hơn chunk_size thì trả về ngay. Nếu không, thử tách bằng separator đầu tiên trong danh sách. Với mỗi phần tách được, nếu cộng vào chunk hiện tại không vượt quá chunk_size thì thêm vào, ngược lại lưu chunk hiện tại và xử lý phần còn lại. Nếu phần tách được vẫn quá dài, đệ quy gọi `_split` với separator tiếp theo.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents`: Lặp qua danh sách Document, tạo embedding cho từng content, lưu vào list `_store` dưới dạng dict với keys: id, content, embedding, metadata. `search`: Tạo embedding cho query, tính dot product với tất cả embeddings trong store, sort theo score giảm dần, trả về top_k kết quả.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter`: Trước tiên filter records theo metadata_filter (so sánh key-value), sau đó gọi `_search_records` trên filtered list. `delete_document`: Dùng list comprehension để giữ lại các records không có doc_id trong metadata, trả về True nếu có record bị xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search()` để lấy top_k chunks liên quan, nối các chunks thành context string, build prompt theo format: "Context:\n{context}\n\nQuestion: {question}", sau đó gọi `llm_fn(prompt)` để lấy câu trả lời.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.0.3, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 16.10s =============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "VF 8 giảm giá 50 triệu" | "Ưu đãi VF 8 là 50.000.000 VNĐ" | high | ~0.85 | Đúng |
| 2 | "Chính sách Mua xe 0 Đồng" | "Chương trình voucher Giờ Trái Đất" | low | ~0.45 | Đúng |
| 3 | "Thu xăng đổi điện hỗ trợ 3%" | "Chuyển đổi xe xăng sang điện được 3%" | high | ~0.82 | Đúng |
| 4 | "Sạc pin miễn phí 10 lần/tháng" | "Bảo hiểm 2 năm cho VF MPV 7" | low | ~0.38 | Đúng |
| 5 | "Lãi suất 5%/năm cho VF 8" | "Lãi suất ưu đãi VF8 là 5%" | high | ~0.88 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Các cặp dự đoán đều khá chính xác. Điều này cho thấy embedding model (text-embedding-3-small) có khả năng nắm bắt ngữ nghĩa semantic tốt, ngay cả khi cách diễn đạt khác nhau miễn là ý nghĩa tương đồng. Đặc biệt, model nhận ra các con số cụ thể (50 triệu, 3%, 5%) là thông tin quan trọng để so sánh.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Chương trình voucher Giờ Trái Đất áp dụng cho khách hàng trong thờ gian nào? | Áp dụng cho khách hàng đặt cọc mua xe trong các giai đoạn 20-22/03/2026 và 26-30/03/2026, đồng thờ xuất hóa đơn đến hết ngày 30/06/2026. |
| 2 | Giá trị voucher dành cho dòng xe VF 8 trong chương trình Giờ Trái Đất là bao nhiêu? | Giá trị voucher của VF 8 là 15.000.000 VNĐ |
| 3 | Chính sách Mua xe 0 Đồng cho phép khách hàng vay tối đa bao nhiêu phần trăm giá trị xe? | Khách hàng được vay tối đa 100% giá trị xe và không cần vốn đối ứng. |
| 4 | Trong chương trình Mãnh liệt vì Tương lai Xanh, khách hàng mua VF 8 được hưởng những ưu đãi gì? | Khách hàng mua VF 8 được chọn một trong hai ưu đãi: giảm 10% MSRP hoặc hỗ trợ lãi suất cố định 5%/năm trong 3 năm đầu. |
| 5 | Chính sách ưu đãi sạc pin áp dụng như thế nào đối với xe mua từ ngày 10/02/2026? | Với xe mua từ ngày 10/02/2026, EC Van và Minio Green được miễn phí 20 lần sạc đầu tiên/xe/tháng tại trụ sạc V-Green đến hết 10/02/2029, còn các dòng xe khác được miễn phí 10 lần sạc đầu tiên/xe/tháng. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Chương trình voucher Giờ Trái Đất | Thông tin đối tượng áp dụng: đặt cọc 20-22/03 và 26-30/03/2026, xuất HĐ đến 30/06/2026 | 0.723 | ✅ Có | Trả về đúng thờ gian áp dụng |
| 2 | Giá trị voucher VF 8 | Bảng giá trị voucher: VF 7/VF 8 = 15.000.000 VNĐ | 0.705 | ✅ Có | Trả về đúng 15 triệu VNĐ |
| 3 | Chính sách Mua xe 0 Đồng | Thông tin cho vay lên tới 100% giá trị xe | 0.692 | ✅ Có | Trả về đúng 100% |
| 4 | Ưu đãi VF 8 chương trình Xanh | Thông tin giảm 10% MSRP cho VF 8, VF 9 | 0.716 | ✅ Có | Trả về đúng 10% MSRP hoặc lãi suất 5% |
| 5 | Chính sách ưu đãi sạc pin | Thông tin ưu đãi sạc đủ 3 năm, áp dụng tới 30/06/2027 | 0.846 | ⚠️ Một phần | Trả về thông tin xe mua TRƯỚC 10/02/2026 thay vì TỪ 10/02/2026 |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

**Nhận xét:**
- Query 5 có vấn đề vì retrieval trả về thông tin về xe mua TRƯỚC 10/02/2026 thay vì TỪ 10/02/2026. Điều này cho thấy cần cải thiện cách diễn đạt query hoặc thêm metadata về thờ gian.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Việc so sánh các chunking strategies giúp tôi nhận ra rằng không có strategy nào là tối ưu cho mọi loại tài liệu. Fixed-size đơn giản nhưng dễ cắt giữa câu, sentence-based giữ ranh giới câu nhưng chunk size không đồng đều. Recursive là sự cân bằng tốt cho tài liệu có cấu trúc.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Metadata filtering rất quan trọng để cải thiện precision của retrieval. Ví dụ lọc theo ngôn ngữ (vi/en) hoặc loại tài liệu (policy/technical) giúp loại bỏ nhiễu và tập trung vào nguồn thông tin phù hợp.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm nhiều metadata hơn như `effective_date` (ngày hiệu lực), `policy_type` (khuyến mãi/bảo hành/sạc pin), và `car_model` (VF 3/5/6/7/8/9). Điều này sẽ giúp trả lời chính xác hơn các câu hỏi liên quan đến thờ gian và đối tượng cụ thể.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **88 / 100** |

**Lý do điểm Results 8/10:** Đưa ra khá đúng ngữ nghĩa trong đa số trường hợp. Query 5 về chính sách sạc pin không trả về chunk chính xác do confusion giữa "từ ngày" và "trước ngày" trong tài liệu.
