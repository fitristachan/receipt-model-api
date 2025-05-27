from ultralytics import YOLO
import re
from typing import List, Dict, Union, Any
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

def normalize_price_typo(text: str) -> str:
    typo_map = {
        'b': '8',
        'B': '8',
        'o': '0',
        'O': '0',
        'l': '1',
        'I': '1',
        'i': '1',
        's': '5',
        'S': '5',
        'z': '2',
        'Z': '2',
    }
    return ''.join(typo_map.get(c, c) for c in text)


def extract_tax_lines_with_context(rec_texts, tax_keywords=None):
    if tax_keywords is None:
        tax_keywords = ["ppn", "pajak", "tax", "pjk", "service", "serv", "pb", "charge", "chrg"]

    tax_lines = []
    tax_amounts = []

    n = len(rec_texts)

    for i, line in enumerate(rec_texts):
        lower_line = line.lower()

        if any(k in lower_line for k in tax_keywords):
            tax_lines.append(line)
            found = False

            # ✅ Cari angka yang terkait keyword di baris ini
            matches = extract_tax_values_from_line_by_keyword(line, tax_keywords)
            tax_amounts.extend(matches)
            found = bool(matches)

            # Kalau belum ketemu, cek baris setelah
            if not found and i + 1 < n:
                next_line = rec_texts[i + 1]
                amount = extract_price_like_from_line(next_line)
                if amount is not None:
                    tax_amounts.append(amount)
                    tax_lines.append(next_line)
                    found = True

            # Kalau masih belum, cek baris sebelum
            if not found and i - 1 >= 0:
                prev_line = rec_texts[i - 1]
                amount = extract_price_like_from_line(prev_line)
                if amount is not None:
                    tax_amounts.append(amount)
                    tax_lines.append(prev_line)

    return tax_lines, tax_amounts


def extract_tax_values_from_line_by_keyword(line, tax_keywords):
    tax_amounts = []
    lower_line = line.lower()

    for keyword in tax_keywords:
        pattern = rf"{keyword}\s*[:=]?\s*([\d.,]+)"
        matches = re.findall(pattern, lower_line)
        for raw_amt in matches:
            if "%" in raw_amt:
                continue
            if is_price_like_number(raw_amt):
                amt = int(raw_amt.replace(",", "").replace(".", ""))
                tax_amounts.append(amt)

    return tax_amounts


def extract_price_like_from_line(line):
    line = normalize_price_typo(line)
    match = re.search(r"([\d.,]+)", line)
    if match:
        raw = match.group(1)
        if "%" in raw:
            return None
        if is_price_like_number(raw):
            return int(raw.replace(",", "").replace(".", ""))
    return None


def is_price_like_number(s: str, min_val: int = 100):
    s_clean = s.replace(",", "").replace(".", "")
    if not s_clean.isdigit():
        return False
    try:
        val = int(s_clean)
        return val >= min_val
    except:
        return False


def clean_item_name(name: str) -> str:
    # Hapus karakter non-alfanumerik di awal dan akhir (selain huruf dan angka)
    return re.sub(r"^[^\w\d]+|[^\w\d]+$", "", name.strip())

def process_receipt_data_with_ner(ner_results: List[Dict], rec_texts: Any) -> Dict:
    """Process receipt data and extract total/tax info from OCR."""
    reconstructed = reconstruct_items(ner_results)
    paired = pair_entities(reconstructed)

    valid_items = []
    discount_keywords = {"uc", "vc", "vc pt", "disc", "voucher", "diskon", "discount"}
    
    i = 0
    n = len(paired)

    while i < n:
        current_item = paired[i]
        current_item['item_name'] = clean_item_name(current_item['item_name'])

        if not current_item['item_name'] or not current_item['price'] or is_noise(current_item['item_name']) or is_short_or_symbol(current_item['item_name']):
            i += 1
            continue

        item_name_lower = current_item['item_name'].lower()

        if i + 1 < n:
            next_item = paired[i + 1]
            next_text = next_item['item_name'].lower()

            if any(keyword in next_text for keyword in discount_keywords):
                current_item['price'] -= next_item['price']
                i += 1

        if isinstance(current_item['price'], (int, float)):
            current_item['price'] = abs(current_item['price'])
        else:
            i += 1
            continue
            
        valid_items.append(current_item)
        i += 1

    # Use original lines (not lowercased) to preserve tax keywords
    tax_lines, tax_amounts = extract_tax_lines_with_context(rec_texts)
    # Print hasil
    print("Baris pajak terdeteksi:")
    for line in tax_lines:
        print(f" - {line}")

    return {
        'status': 'success',
        'tax': sum(tax_amounts),
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

def read_image(image, ocr):
    preprocessed = preprocess(image)
    boxes, classes = process_model(preprocessed)

    print("Langsung cek keyword dari OCR global...")

    preprocessed_ocr = preprocess_for_ocr(preprocessed)

    # Gunakan OCR global
    ocr_global = ocr.predict(preprocessed_ocr)
    
    # If ocr_result is a list containing one dict
    if isinstance(ocr_global, list) and len(ocr_global) > 0:
        ocr_global = ocr_global[0]
    
    # Now access rec_texts
    rec_texts = ocr_global['rec_texts']
    detected_global_lower = " ".join([t.lower() for t in rec_texts if t.strip()])

    detected_lines = []
    
    if any(keyword in detected_global_lower for keyword in keywords):
        print("Keyword ditemukan. Proses ekstraksi data receipt...")

        for idx, bbox in enumerate(boxes):
            try:
                cropped_image = crop_image_by_bbox(preprocessed_ocr, bbox)
                ocr_result = ocr.predict(cropped_image)
                if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
                    ocr_result = ocr_result[0]  # Get the first (and usually only) result dict
                    
                    # Extract text line by line
                    for text in ocr_result['rec_texts']:
                        if text.strip():  # Skip empty lines
                            detected_lines.append(text)  # Append the recognized text
                
                    # (Optional) Get bounding boxes for each line (if needed)
                    # line_bboxes = ocr_result['dt_polys']  # List of bounding boxes (if required)

            except ValueError as e:
                print(f"Error dalam cropping gambar: {e}")
                continue

        detected_text_lower = " ".join([t.lower() for t in detected_lines])
        print(detected_text_lower)

        lines = split_detected_text(detected_text_lower)
        ner_results = detect_class(lines)
        final_result = process_receipt_data_with_ner(ner_results, rec_texts)

        print("SPLIT lines:", lines)
        print("NER results:", ner_results)
        print("FINAL:", final_result)

        return final_result