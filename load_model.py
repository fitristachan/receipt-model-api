from ultralytics import YOLO
import re
from typing import List, Dict, Union
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


import re
from typing import List, Dict, Union

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
        'r0unding','disk0n','disc0unt', 'vc', 'vc pt'
        'rounding', 'diskon', 'discount', 'total', 'grand total', 'change', 'kembalian', 'srand tl'
    ]
    text = text.lower()
    return any(keyword in text for keyword in noise_keywords)


def is_short_or_symbol(text: str, min_length=3) -> bool:
    """Check if text is too short or just symbols/numbers."""
    stripped = text.strip()
    # Skip if:
    # - Length < min_length (default: 3)
    # - Only digits/symbols (e.g., "1", "A", ",M")
    return (
        len(stripped) < min_length or
        stripped.isdigit() or
        (len(stripped) == 1 and not stripped.isalpha())
    )


def clean_item_name(name: str) -> str:
    # Hapus karakter non-alfanumerik di awal dan akhir (selain huruf dan angka)
    return re.sub(r"^[^\w\d]+|[^\w\d]+$", "", name.strip())

def process_receipt_data_with_ner(ner_results: List[Dict]) -> Dict:
    """Process receipt data with discount handling."""
    reconstructed = reconstruct_items(ner_results)
    paired = pair_entities(reconstructed)
    
    valid_items = []
    discount_keywords = {"uc", "vc", "vc pt", "disc", "voucher", "diskon", "discount"}  # Keywords that indicate discounts
    
    i = 0
    n = len(paired)
    
    while i < n:
        current_item = paired[i]
        
        # Bersihkan item name
        current_item['item_name'] = clean_item_name(current_item['item_name'])
        
        
        # Skip if invalid
        if not current_item['item_name'] or not current_item['price'] or is_noise(current_item['item_name']) or is_short_or_symbol(current_item['item_name']):
            i += 1
            continue

        # Check if the NEXT item is a discount
        if i + 1 < n:
            next_item = paired[i + 1]
            next_text = next_item['item_name'].lower()
            
            # If next item is a discount, subtract its price
            if any(keyword in next_text for keyword in discount_keywords):
                current_item['price'] -= next_item['price']
                i += 1  # Skip the discount item
                
        valid_items.append(current_item)
        i += 1
    
    return {
        'status': 'success',
        'items': valid_items
    }

# Inisialisasi OCR
keywords = ['total', 'subtotal', 'amount', 'jumlah', 't0tal', 'subt0tal']

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

    # Gunakan OCR global
    ocr_result = ocr.predict(preprocessed_ocr)
    
    # If ocr_result is a list containing one dict
    if isinstance(ocr_result, list) and len(ocr_result) > 0:
        ocr_result = ocr_result[0]
    
    # Now access rec_texts
    detected_lines = ocr_result['rec_texts']
    # print(f"detected_lines: {detected_lines}")
    detected_text_lower = " ".join([t.lower() for t in detected_lines if t.strip()])
    print(f"detected_text_lower: {detected_text_lower}")
    
    if any(keyword in detected_text_lower for keyword in keywords):
        print("Keyword ditemukan. Proses ekstraksi data receipt...")

        # OCR berdasarkan bbox dari hasil deteksi model
        detected_lines = []

        for idx, bbox in enumerate(boxes):
            try:
                cropped_image = crop_image_by_bbox(preprocessed_ocr, bbox)
                ocr_result = ocr.predict(cropped_image)
                if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
                    ocr_result = ocr_result[0]  # Get the first (and usually only) result dict
                    
                    # Extract text line by line
                    for text in ocr_result['rec_texts']:
                        if text.strip():  # Skip empty lines
                            detected_lines.append(text) 

            except ValueError as e:
                print(f"Error dalam cropping gambar: {e}")
                continue

        detected_text_lower = " ".join([t.lower() for t in detected_lines])
        print(detected_text_lower)

        lines = split_detected_text(detected_text_lower)
        ner_results = detect_class(lines)
        final_result = process_receipt_data_with_ner(ner_results)

        print("SPLIT lines:", lines)
        print("NER results:", ner_results)
        print("FINAL:", final_result)

        return final_result

    print("Tidak ada keyword penting ditemukan. Gambar bukan receipt/invoice.")
    return {
        "status": "not_receipt_invoice",
        "error_msg": "Gambar yang Anda masukkan bukan receipt/invoice."
    }
