from ultralytics import YOLO
import re
from typing import List, Dict, Union
import easyocr
import numpy as np
import cv2
import os
from preprocessing import preprocess, preprocess_for_ocr

##LOAD MODEL
# Dapatkan direktori dari file Python saat ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Bangun path absolut ke model
model_path = os.path.join(BASE_DIR, 'best.pt')

# Muat model
model = YOLO(model_path)

def process_model(image):
    image_resized = preprocess(image)
    results = model(image_resized)[0]  # Ambil hasil pertama
    boxes = results.boxes  # Bounding box predictions

    bboxes = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    classes = boxes.cls.cpu().numpy().astype(int)  # Class index (int)

    return bboxes, classes

##DEFINE LABEL
label_map = {
    0: 'table',
    1: 'not_table'
}

##SPLIT TEXT
def split_detected_text(detected_text: str) -> List[str]:
    """
    Memecah string panjang hasil OCR menjadi list baris berdasarkan pola 'text price'
    Misalnya: "nasi campur 75,ooo ayam 60,ooo" → ["nasi campur 75,ooo", "ayam 60,ooo"]
    """

    # Cari semua potongan yang diakhiri angka (misal 75,000 / 60.000 / 30.0)
    # Asumsikan bahwa item akan memiliki angka di akhir
    pattern = re.compile(r'(.*?\d[\d.,]*)(?=\s+[a-zA-Z]|$)')
    lines = pattern.findall(detected_text)

    # Bersihkan trailing whitespaces
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def extract_multiple_items_from_line(line: str) -> List[Dict[str, Union[str, int]]]:
    """
    Ekstrak pasangan item dan harga dari satu baris teks OCR.
    Menangani multiple item+price dalam satu baris dengan memanfaatkan regex dan heuristik teks sebelumnya.
    """

    # Cari semua harga di baris (format bebas: 3300, 3,300, 3.300, dst)
    matches = list(re.finditer(r'(\d{1,3}(?:[.,]\d{3})+|\d{4,})', line))

    if not matches:
        return []

    items = []
    last_end = 0

    for match in matches:
        price_str = match.group()
        try:
            price = int(price_str.replace(",", "").replace(".", ""))
        except:
            continue

        # Ambil teks sebelum harga sebagai item
        item_text = line[last_end:match.start()].strip()

        # Hindari item kosong atau angka doang
        if item_text and not item_text.replace(" ", "").isdigit():
            items.append({
                "item_name": item_text,
                "price": price
            })

        last_end = match.end()

    return items


def clean_ocr_text(line: str) -> str:
    # Gabungkan angka yang terpisah spasi: "10 0o0" -> "10000"
    line = re.sub(r'(\d)\s+(\d)', r'\1\2', line)

    # Ganti huruf 'o' atau 'O' yang tertulis sebagai nol
    line = line.replace('o', '0').replace('O', '0')

    return line

def detect_class(lines: List[str]) -> List[Dict]:
    processed_results = []

    for line in lines:
        item_price_pairs = extract_multiple_items_from_line(line)
        if item_price_pairs:
            for pair in item_price_pairs:
                processed_results.append({
                    'class': 'item',
                    'text': pair['item_name']  # ← Fix di sini
                })
                processed_results.append({
                    'class': 'price',
                    'price': pair['price']
                })
        else:
            processed_results.append({
                'class': 'item',
                'text': line
            })

    return processed_results


def reconstruct_items(ner_results: List[Dict]) -> List[Dict]:
    reconstructed = []
    current_item = []
    last_was_item = False
    
    for entry in ner_results:
        if entry['class'] == 'item':
            if not last_was_item and current_item:
                # Jika bertemu item baru, selesaikan item sebelumnya
                reconstructed.append({
                    'class': 'item',
                    'text': ' '.join(current_item).strip(),
                    'original_parts': current_item.copy()
                })
                current_item = []
            current_item.append(entry['text'])
            last_was_item = True
        else:
            if current_item:
                # Selesaikan item yang sedang dibangun
                reconstructed.append({
                    'class': 'item',
                    'text': ' '.join(current_item).strip(),
                    'original_parts': current_item.copy()
                })
                current_item = []
            reconstructed.append(entry)  # Tambahkan price/entitas lain
            last_was_item = False
    
    # Tambahkan sisa item jika ada
    if current_item:
        reconstructed.append({
            'class': 'item',
            'text': ' '.join(current_item).strip(),
            'original_parts': current_item.copy()
        })
    
    return reconstructed

def pair_entities(ner_results: List[Dict]) -> List[Dict]:
    """Memasangkan item dan price yang berurutan"""
    paired = []
    i = 0
    n = len(ner_results)
    
    while i < n:
        if ner_results[i]['class'] == 'item':
            item = ner_results[i]['text']
            price = None
            numeric_price = None

            # Cari price berikutnya
            if i + 1 < n and ner_results[i+1]['class'] == 'price':
                price_entry = ner_results[i+1]
                price = price_entry.get('price')  # Akses 'price' jika sudah diganti
                numeric_price = price_entry.get('price')  # Atau akses langsung numeric value

                i += 1  # Lewati price yang sudah diproses
            
            paired.append({
                'item_name': item,
                'price': numeric_price
            })
        i += 1
    
    return paired


def is_noise(text: str) -> bool:
    """
    Mengecek apakah suatu teks termasuk 'noise' (bukan item menu, misalnya total, service, dll)
    """
    noise_keywords = [
        'subtotal', 'sub total', 'service', 'tax', 'pajak', 'pb1', 't0tal', 'subt0tal',
        'r0unding','disk0n','disc0unt',
        'rounding', 'diskon', 'discount', 'total', 'grand total', 'change', 'kembalian', 'srand tl'
    ]
    text = text.lower()
    return any(keyword in text for keyword in noise_keywords)


def process_receipt_data_with_ner(ner_results: List[Dict]) -> Dict:
    """Pipeline utama dengan NER"""
    # 1. Rekonstruksi item multi-kata
    reconstructed = reconstruct_items(ner_results)
    
    # 2. Pasangkan item dengan price
    paired = pair_entities(reconstructed)
    
    # 3. Filter hasil
    valid_items = [
        item for item in paired 
        if item['item_name'] and item['price'] and not is_noise(item['item_name'])
    ]
    
    return {
        'status': 'success',
        'items': valid_items
    }


# Inisialisasi
keywords = ['total', 'subtotal', 'amount', 'jumlah', 't0tal', 'subt0tal']
reader = easyocr.Reader(['en', 'id']) 

def crop_image_by_bbox(image: np.ndarray, bbox: list) -> np.ndarray:
    height, width = image.shape[:2]

    if len(bbox) == 4:
        x_min, y_min, x_max, y_max = bbox
        x_min = int(max(0, min(x_min, width - 1)))
        x_max = int(max(0, min(x_max, width - 1)))
        y_min = int(max(0, min(y_min, height - 1)))
        y_max = int(max(0, min(y_max, height - 1)))

        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image

    elif len(bbox) == 8:
        pts = np.array(bbox, dtype=np.float32).reshape(-1, 2)
        dst_pts = np.array([[0, 0], [639, 0], [639, 639], [0, 639]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(pts, dst_pts)
        cropped_image = cv2.warpPerspective(image, matrix, (640, 640))
        return cropped_image

    else:
        raise ValueError("Format bbox tidak valid.")

    
def read_image(image):
    preprocessed = preprocess(image)

    boxes, classes = process_model(preprocessed)

    print("Langsung cek keyword dari OCR global...")

    preprocessed_ocr = preprocess_for_ocr(preprocessed)
    ocr_result = reader.readtext(preprocessed_ocr)
    detected_lines = [text for _, text, _ in ocr_result]
    detected_text_lower = " ".join([t.lower() for t in detected_lines])
    # print(f"cek keyword: {detected_text_lower}")

    # Cek apakah ada keyword di teks hasil OCR
    if any(keyword in detected_text_lower for keyword in keywords):
        print("Keyword ditemukan. Proses ekstraksi data receipt...")

        #OCR berdasarkan bbox dari hasil deteksi model
        detected_lines = []
        
        for idx, bbox in enumerate(boxes):
            try:
                cropped_image = crop_image_by_bbox(preprocessed_ocr, bbox)
                ocr_result = reader.readtext(cropped_image)
        
                # Ambil teks saja dari hasil OCR
                detected_lines.extend([text for _, text, _ in ocr_result])
        
            except ValueError as e:
                print(f"Error dalam cropping gambar: {e}")
                continue

        detected_text_lower = " ".join([t.lower() for t in detected_lines])
        print(detected_text_lower)
        
        lines = split_detected_text(detected_text_lower)
        ner_results = detect_class(lines)
        final_result = process_receipt_data_with_ner(ner_results)
        
        # Output
        print("SPLIT lines:", lines)
        print("NER results:", ner_results)
        print("FINAL:", final_result)

        return final_result

    print("Tidak ada keyword penting ditemukan. Gambar bukan receipt/invoice.")
    return {
        "status": "not_receipt_invoice",
        "error_msg": "Gambar yang Anda masukkan bukan receipt/invoice."
    }