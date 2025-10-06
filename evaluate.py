from dotenv import load_dotenv
import json
import os
from tqdm import tqdm # Thư viện để tạo thanh tiến trình (progress bar)

# Tải các biến môi trường từ file .env
load_dotenv(override=True)

# Import các thành phần cần thiết từ hệ thống của bạn
from app.core.tool_suggestion import ToolSuggestion
from app.core.tool_suggestion.constant import COLLECTION_NAME, CHROMADB_DIR
from app.core.tool_suggestion import utils as chroma_utils

def load_benchmark_data(filepath: str) -> list:
    """Tải dữ liệu benchmark từ file JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file benchmark tại '{filepath}'")
        return []
    except json.JSONDecodeError:
        print(f"Lỗi: File '{filepath}' không phải là một file JSON hợp lệ.")
        return []

def get_all_tool_ids() -> list:
    """Lấy danh sách ID của tất cả các tool có trong hệ thống."""
    try:
        collection = chroma_utils.get_collection(name=COLLECTION_NAME, path=CHROMADB_DIR)
        if not collection:
            return []
        
        all_res = collection.get()
        # Dùng set để lấy các tool_id duy nhất và chuyển lại thành list
        all_ids = {meta["tool_id"] for meta in all_res["metadatas"]}
        return list(all_ids)
    except Exception as e:
        print(f"Lỗi khi lấy danh sách tool IDs: {e}")
        return []


def run_evaluation(benchmark_data: list, all_tool_ids: list):
    """
    Chạy quá trình đánh giá trên bộ dữ liệu benchmark.
    """
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    
    results_log = []

    print("Bắt đầu quá trình đánh giá...")
    # Sử dụng tqdm để hiển thị thanh tiến trình
    for item in tqdm(benchmark_data, desc="Đang đánh giá"):
        prompt = item["prompt"]
        expected_ids = set(item["expected_tool_ids"]) # Chuyển sang set để xử lý dễ dàng

        # Lấy kết quả dự đoán từ hệ thống
        suggestions = ToolSuggestion.suggest(prompt, all_tool_ids, [])
        predicted_ids = {choice.tool_id for choice in suggestions.tool_choices}

        # Tính toán các chỉ số
        tp = len(expected_ids.intersection(predicted_ids))
        fp = len(predicted_ids.difference(expected_ids))
        fn = len(expected_ids.difference(predicted_ids))

        # Cộng dồn vào tổng
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Lưu lại log chi tiết để phân tích lỗi
        results_log.append({
            "prompt": prompt,
            "expected": sorted(list(expected_ids)),
            "predicted": sorted(list(predicted_ids)),
            "is_correct": fp == 0 and fn == 0,
            "fp_ids": sorted(list(predicted_ids.difference(expected_ids))),
            "fn_ids": sorted(list(expected_ids.difference(predicted_ids)))
        })

    # Tính toán các chỉ số tổng thể
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    summary = {
        "total_prompts": len(benchmark_data),
        "correct_predictions": sum(1 for log in results_log if log["is_correct"]),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn
    }
    
    return summary, results_log

def print_report(summary: dict, log: list):
    """In báo cáo kết quả ra màn hình."""
    
    print("\n" + "="*50)
    print("   BÁO CÁO KẾT QUẢ ĐÁNH GIÁ")
    print("="*50)
    
    correct_percent = (summary['correct_predictions'] / summary['total_prompts']) * 100 if summary['total_prompts'] > 0 else 0
    print(f"Tổng số prompt: {summary['total_prompts']}")
    print(f"Số dự đoán đúng hoàn toàn: {summary['correct_predictions']} ({correct_percent:.2f}%)")
    
    print("\n--- CHỈ SỐ TỔNG THỂ ---")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall:    {summary['recall']:.4f}")
    print(f"F1-Score:  {summary['f1_score']:.4f}")
    
    print("\n" + "-"*50)
    print("   PHÂN TÍCH LỖI CHI TIẾT")
    print("-"*50)

    false_positives = [item for item in log if item['fp_ids']]
    false_negatives = [item for item in log if item['fn_ids']]

    if not false_positives and not false_negatives:
        print("Không có lỗi nào được tìm thấy.")
        return

    if false_positives:
        print("\nSố lượng False Positives: ", len(false_positives))
        print("\n[!] CÁC TRƯỜNG HỢP DỰ ĐOÁN THỪA (False Positives):")
        for item in false_positives:
            print(f"  - Prompt:    '{item['prompt']}'")
            print(f"    Dự đoán thừa ID: {item['fp_ids']}")
            print(f"    (Đáp án đúng: {item['expected']})")

    if false_negatives:
        print("\nSố lượng False Negatives: ", len(false_negatives))
        print("\n[!] CÁC TRƯỜNG HỢP DỰ ĐOÁN THIẾU (False Negatives):")
        for item in false_negatives:
            print(f"  - Prompt:    '{item['prompt']}'")
            print(f"    Bỏ sót ID: {item['fn_ids']}")
            print(f"    (Hệ thống dự đoán: {item['predicted']})")

if __name__ == "__main__":
    benchmark_file_path = 'benchmark.json'
    
    print("Đang tải dữ liệu benchmark...")
    benchmark = load_benchmark_data(benchmark_file_path)
    
    if benchmark:
        print("Đang lấy danh sách tất cả các tool ID...")
        all_tool_ids = get_all_tool_ids()
        
        if all_tool_ids:
            summary_report, detailed_log = run_evaluation(benchmark, all_tool_ids)
            print_report(summary_report, detailed_log)
        else:
            print("Không thể lấy được danh sách tool. Dừng quá trình đánh giá.")
