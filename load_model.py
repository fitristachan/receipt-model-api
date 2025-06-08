from ultralytics import YOLO
import re
from typing import List, Dict, Union, Any, Tuple
import numpy as np
import cv2
import os
import requests
from io import BytesIO
from preprocessing import preprocess, preprocess_for_ocr

##LOAD MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best.pt') 
model = YOLO(model_path)

def process_model(image: np.ndarray) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    image_resized = preprocess(image)
    results = model(image_resized)[0]
    if results.boxes is None or results.boxes.cls is None or len(results.boxes.cls) == 0:
        return None, None
    return results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy().astype(int)

# --- KEYWORDS---
TAX_KEYWORDS = {"ppn", "pajak", "tax", "pjk", "service", "serv", "pb1", "pb 1", "charge", "chrg"}
RECEIPT_DISCOUNT_KEYWORDS = {"uc", "vc", "vc pt", "disc", "voucher", "diskon", "discount", "potongan"}

def clean_digits_for_int(price_str_candidate: str) -> Union[int, None]:
    if not price_str_candidate: return None
    cleaned_str = re.sub(r'^RP\s*\.?\s*', '', price_str_candidate, flags=re.IGNORECASE).strip()
    cleaned_str = cleaned_str.replace(':', '.').replace('o', '0',-1).replace('O', '0',-1).replace('ộ', '0',-1)
    cleaned_str = re.sub(r'(\d)\s+(\d)', r'\1\2', cleaned_str)
    just_digits = re.sub(r'[.,]', '', cleaned_str)
    if just_digits.isdigit():
        try: return int(just_digits)
        except ValueError: return None
    return None

def extract_discount_info_from_line(line: str, discount_keywords: set) -> Union[Dict[str, Any], None]:
    line_lower = line.lower()
    has_discount_keyword = any(keyword in line_lower for keyword in discount_keywords)
    
    match_paren = re.search(r'\(\s*([\d,.]+)\s*\)', line)
    if match_paren:
        amount = clean_digits_for_int(match_paren.group(1))
        if amount and amount > 0:
            desc = line[:match_paren.start()].strip() or "Discount"
            return {'text': desc.upper(), 'amount': amount, 'type': 'parentheses'}

    match_negative = re.search(r'-\s*([\d,.]+)', line)
    if match_negative:
        amount = clean_digits_for_int(match_negative.group(1))
        if amount and amount > 0:
            desc = line[:match_negative.start()].strip() or "Discount"
            return {'text': desc.upper(), 'amount': amount, 'type': 'negative'}

    if has_discount_keyword:
        for keyword in discount_keywords:
            pattern = rf"{re.escape(keyword)}\s*[:=\-]?\s*(?:(?:RP|Rp)\.?\s*)?([\d,.]+)"
            keyword_match_amount = re.search(pattern, line, re.IGNORECASE)
            if keyword_match_amount:
                amount = clean_digits_for_int(keyword_match_amount.group(1))
                if amount and amount > 0:
                    desc = line[:keyword_match_amount.start()].strip() or keyword.upper()
                    return {'text': desc.upper() or keyword.upper(), 'amount': amount, 'type': 'keyword'}
        
        trailing_numbers = re.findall(r'([\d,.]+)\b', line)
        if trailing_numbers:
            for num_str in reversed(trailing_numbers):
                amount = clean_digits_for_int(num_str)
                if amount and amount > 0 and amount < 1000000:
                    desc = line.replace(num_str, "").strip() or "Discount"
                    return {'text': desc.upper(), 'amount': amount, 'type': 'keyword_general_number'}
    return None

def _clean_price_for_extraction_v2(text_price: str) -> Union[int, None]:
    if not text_price: return None
    text_cleaned = text_price.replace('/', '') 
    text_cleaned = re.sub(r"[^\d]", "", text_cleaned)
    if text_cleaned.isdigit() and text_cleaned:
        try:
            return int(text_cleaned)
        except ValueError:
            return None
    return None

def clean_name_general(name: str) -> str:
    if not name: return ""
    name = re.sub(r"[^\w\s&/\-\(\)\%\+]", "", name) 
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name.title()

def extract_name_and_price_from_line_v2(line: str) -> Tuple[Union[str, None], Union[int, None]]:
    line = line.strip()
    if not line:
        return None, None

    match = re.match(r'^(.*?)\s+([\d.,/]+)$', line)
    
    name_part = None
    price_val = None

    if match:
        name_candidate = match.group(1).strip()
        price_str_candidate = match.group(2).strip()
        price = _clean_price_for_extraction_v2(price_str_candidate)
        
        if price is not None:
            name_part = clean_name_general(name_candidate)
            price_val = price
            if name_part:
                qty_ending_match = re.match(r'^(.*?)\s+(\d{1,2})$', name_part) # Cari angka 1-2 digit di akhir nama
                if qty_ending_match and price_val is not None and price_val >= 1000 : 
                    name_before_qty = clean_name_general(qty_ending_match.group(1))
                    if name_before_qty:
                        name_part = name_before_qty
                        # quantity_extracted = qty_ending_match.group(2) # Bisa disimpan jika perlu
        else: 
            name_part = clean_name_general(line)
            price_val = None
    else: 
        if any(c.isalpha() for c in line):
            name_part = clean_name_general(line)
        price_val = None

    return name_part if name_part else None, price_val


def extract_ocr_items_v2(text: str) -> Dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items = []
    
    i = 0
    while i < len(lines):
        current_line = lines[i]
        print(f"DEBUG v2 Processing line: '{current_line}'")

        discount_info = extract_discount_info_from_line(current_line, RECEIPT_DISCOUNT_KEYWORDS)
        
        if discount_info and discount_info.get('amount'):
            discount_amount = discount_info['amount']
            print(f"DEBUG v2: Found discount: {discount_amount} from line '{current_line}' (type: {discount_info.get('type')})")
            
            if items: 
                last_item = items[-1]
                print(f"DEBUG v2: Applying discount {discount_amount} to PREVIOUS item '{last_item['item_name']}' (current price: {last_item['price']})")
                last_item['price'] -= discount_amount
                if last_item['price'] < 0:
                    last_item['price'] = 0
            else:
                print(f"DEBUG v2: Warning - Found discount '{current_line}' but no previous item to apply it to.")
            i += 1
            continue
        
        name, price_on_line = extract_name_and_price_from_line_v2(current_line)
        print(f"DEBUG v2: extract_name_and_price_from_line_v2('{current_line}') -> name='{name}', price='{price_on_line}'")

        if name and price_on_line is not None:
            items.append({"item_name": name, "price": price_on_line})
            
        elif name and price_on_line is None: 
            print(f"DEBUG v2: Line with name but no price, skipping as main item (could be desc/header): '{name}'")
        
        else: 
            print(f"DEBUG v2: Line did not yield name/price or was discarded: '{current_line}'")

        i += 1

    return {
        "items": items,
        "tax": 0, # Pajak akan dihitung oleh extract_tax_lines_with_context secara global
        "status": "success" if items else "no_items_extracted"
    }

def old_is_noise(text: str) -> bool:
    noise_keywords = ['subtotal', 'sub total', 'service', 'tax', 'pajak', 'pb1', 't0tal', 'subt0tal','r0unding','disk0n','disc0unt', 'vc', 'vc pt', 'rounding', 'diskon', 'discount', 'total', 'grand total', 'change', 'kembalian', 'srand tl', 'menu', 'price', 'jumlah', 'kembali', 'tagihan']
    text_lower = text.lower()
    if text_lower == "vc" and text_lower in RECEIPT_DISCOUNT_KEYWORDS: return False 
    return any(keyword in text_lower for keyword in noise_keywords)

def extract_tax_values_from_line_by_keyword(line_text: str, tax_keywords_list: List[str]) -> List[int]:
    tax_amounts = []
    for keyword in tax_keywords_list:
        pattern_flexible = rf"(?:^|\s){re.escape(keyword)}(?:\s*[:=\-]?\s*|\s+)(?:(?:RP|Rp)\.?\s*)?([\d.,]+)\b"
        pattern_stricter_keyword = rf"{re.escape(keyword)}\s*[:=\-]?\s*(?:(?:RP|Rp)\.?\s*)?([\d.,]+)\b"
        matches = []
        for pattern in [pattern_flexible, pattern_stricter_keyword]:
            try:
                current_matches = re.findall(pattern, line_text, flags=re.IGNORECASE)
                matches.extend(current_matches)
            except re.error as e:
                print(f"Regex error in extract_tax_values_from_line_by_keyword for pattern {pattern}: {e}")
        for raw_amount in matches:
            price_part = raw_amount if isinstance(raw_amount, str) else raw_amount[-1]
            if "%" in price_part: continue
            amount = clean_digits_for_int(price_part)
            if amount is not None and amount > 0 : 
                tax_amounts.append(amount)
    return list(set(tax_amounts))

def extract_tax_lines_with_context(rec_texts: List[str], tax_keywords_list: List[str] = None) -> Tuple[List[str], List[int]]:
    if tax_keywords_list is None: tax_keywords_list = list(TAX_KEYWORDS)
    all_tax_amounts = []; detected_tax_line_texts = []
    for i, line_text_iter in enumerate(rec_texts):
        line_text_lower = line_text_iter.lower()
        keyword_present = False
        for kw in tax_keywords_list:
            if re.search(r'\b' + re.escape(kw) + r'\b', line_text_lower):
                keyword_present = True
                break
        if keyword_present:
            amounts_from_line = extract_tax_values_from_line_by_keyword(line_text_iter, tax_keywords_list)
            if amounts_from_line:
                print(f"DEBUG Tax: Found tax value(s) {amounts_from_line} on keyword line: '{line_text_iter}'")
                all_tax_amounts.extend(amounts_from_line)
                detected_tax_line_texts.append(line_text_iter)
            elif i + 1 < len(rec_texts):
                next_line_text = rec_texts[i+1].strip()
                price_in_next_line = clean_digits_for_int(next_line_text)
                next_line_is_also_tax_keyword = False
                for kw_next in tax_keywords_list:
                    if re.search(r'\b' + re.escape(kw_next) + r'\b', next_line_text.lower()):
                        next_line_is_also_tax_keyword = True
                        break
                if price_in_next_line and price_in_next_line > 0 and not next_line_is_also_tax_keyword:
                    if not any(noise_kw in line_text_lower for noise_kw in ['subtotal', 'total', 'tagihan', 'bayar', 'grand total']):
                        print(f"DEBUG Tax: Found tax value {price_in_next_line} on line AFTER keyword line: '{line_text_iter}' -> '{next_line_text}'")
                        all_tax_amounts.append(price_in_next_line)
                        detected_tax_line_texts.append(f"{line_text_iter} -> {next_line_text}")
    return list(set(detected_tax_line_texts)), list(set(all_tax_amounts)) # Unikkan juga pajaknya


def parse_ocr_space_table_for_invoice(ocr_json_response: Dict) -> List[Dict]:
    parsed_items = [];
    if not ocr_json_response or not ocr_json_response.get("ParsedResults"): return parsed_items
    result = ocr_json_response["ParsedResults"][0]
    if result.get("FileParseExitCode") not in [1, 2] : return parsed_items
    if not result.get("TextTable"): return parsed_items
    text_table = result["TextTable"]; rows = text_table.get("TextTableRows", [])
    for row_data in rows:
        cells = row_data.get("TextTableCells", [])
        if len(cells) < 2: continue
        item_name_str = cells[0].get("Text", "").strip() if len(cells) > 0 else ""
        raw_price_str = "";
        if len(cells) >= 4: raw_price_str = cells[3].get("Text", "").strip()
        elif len(cells) >= 2: raw_price_str = cells[len(cells)-1].get("Text", "").strip()
        price_val = clean_digits_for_int(raw_price_str) 
        if item_name_str and price_val is not None and price_val > 0:
            lower_item_name = item_name_str.lower()
            skip_invoice_item_keywords = ["keterangan", "description", "harga", "price", "jml", "qty", "total", "amount", 
                                          "sub total", "pajak", "pembayaran", "kepada :", "no invoice :", "tanggal :", "item", "no.", "uraian"]
            if any(kw in lower_item_name for kw in skip_invoice_item_keywords):
                if len(item_name_str.split()) <= 2 and any(kw == lower_item_name for kw in ["keterangan", "harga", "jml", "total", "item", "price", "qty", "amount", "no.", "uraian"]): 
                    continue
            parsed_items.append({"item_name": item_name_str.upper(), "price": price_val}) 
    return parsed_items

def parse_invoice_lines_from_text(invoice_text_lines: List[str]) -> List[Dict]:
    parsed_items = []
    for line_idx, line in enumerate(invoice_text_lines):
        line_lower = line.lower()
        skip_keywords = ["kepada yth", "yth.", "no. faktur", "faktur penjualan", 
                         "kepada :", "tanggal :", "no invoice :", "pembayaran :", 
                         "sub total", "total rp", "pajak rp", "ppn rp", "grand total",
                         "terimakasih atas", "bank :", "no. rek :", "a/n :", "jumlah total", "jatuh tempo"]
        if any(kw in line_lower for kw in skip_keywords) or \
           line_lower in ["keterangan", "harga", "jml", "total", "deskripsi", "jumlah", "item", "description", "amount", "price", "qty", "uraian", "satuan"]: 
            continue
        
        name_part, price_val = extract_name_and_price_from_line_v2(line) 

        if name_part and price_val is not None and price_val > 0: 
            if len(name_part) > 1 and not name_part.isdigit() and \
               not (len(name_part.split()) == 1 and name_part.lower() in ["rp", "idr"]):
                parsed_items.append({"item_name": name_part.upper(), "price": price_val}) # Nama sudah di .title() oleh extract_name_and_price_from_line_v2
    return parsed_items

# --- FUNGSI PEMBERSIH NAMA ITEM YANG TELAH DIPERBARUI ---
def final_item_name_cleaning(item_name: str) -> str:
    """
    Membersihkan nama item dengan menghapus pola angka dan teks di akhir string yang
    kemungkinan merupakan sisa dari harga, kuantitas, atau format invoice.

    Args:
        item_name: Nama item asli.

    Returns:
        Nama item yang sudah dibersihkan.
    """
    if not item_name:
        return ""

    # Pola 0 (BARU): Menghapus pola spesifik dari invoice.
    # Contoh: "KAOS RP 100000 1 RP" -> "KAOS"
    # Pola ini mencari " RP <harga> <kuantitas> RP" di akhir string.
    # Menggunakan re.IGNORECASE untuk menangani "RP" atau "rp".
    cleaned = re.sub(r'\s+RP\s+\d+\s+\d+\s+RP$', '', item_name, flags=re.IGNORECASE).strip()
    if cleaned != item_name:
        return cleaned

    # Pola 1: Menghapus "<spasi><1-2 digit><spasi><3+ digit>" dari akhir.
    # Contoh: "Idm Tas Rmh Lngk Kcl 1 3100" -> "Idm Tas Rmh Lngk Kcl"
    cleaned = re.sub(r'\s+\d{1,2}\s+\d{3,}$', '', item_name).strip()
    if cleaned != item_name:
        return cleaned

    # Pola 2: Menghapus "<spasi><3+ digit>" dari akhir.
    # Contoh: "Some Item 45000" -> "Some Item"
    cleaned = re.sub(r'\s+\d{3,}$', '', item_name).strip()
    if cleaned != item_name:
        return cleaned
        
    # Pola 3: Menghapus kuantitas "<spasi><1-2 digit>" dari akhir.
    # Contoh: "Bambi Baby Powder 1" -> "Bambi Baby Powder"
    match = re.match(r'^(.*[a-zA-Z].*)\s(\d{1,2})$', item_name)
    if match:
        return match.group(1).strip()

    return item_name # Kembalikan nama asli jika tidak ada pola yang cocok


def process_data_custom(
    parsed_items_list: List[Dict], 
    all_rec_texts_from_ocr: List[str],
    is_receipt_logic: bool = True,
    pre_calculated_tax: Union[int, None] = None
) -> Dict:
    valid_items_final = []
    print(f"DEBUG process_data_custom: Input items = {parsed_items_list}")
    print(f"DEBUG process_data_custom: is_receipt_logic = {is_receipt_logic}, pre_calculated_tax = {pre_calculated_tax}")
    unique_items_tracker = set()

    for item_data in parsed_items_list:
        item_name_original = item_data.get('item_name', "")
        price = item_data.get('price')
        if not item_name_original or price is None: continue
        
        # Terapkan pembersihan akhir pada nama item
        item_name_final = final_item_name_cleaning(item_name_original)
        
        min_len = 2
        if len(item_name_final) < min_len: continue
        if item_name_final.isdigit() and not (item_name_final.lower() == "vc" and is_receipt_logic) : continue
        if old_is_noise(item_name_final) and not (item_name_final.lower() == "vc" and is_receipt_logic): continue
        if price < 0: price = 0

        item_signature = (item_name_final.lower(), price)
        if item_signature in unique_items_tracker:
            print(f"DEBUG process_data_custom: Skipping duplicate item: {item_signature}")
            continue
        unique_items_tracker.add(item_signature)
        valid_items_final.append({'item_name': item_name_final, 'price': price})
    
    total_tax_final = 0
    if is_receipt_logic and pre_calculated_tax is not None:
        total_tax_final = pre_calculated_tax
        print(f"DEBUG process_data_custom: Using pre_calculated_tax for receipt: {total_tax_final}")
    else: 
        print(f"DEBUG process_data_custom: Calculating tax using extract_tax_lines_with_context on global OCR text.")
        _, tax_amounts_detected = extract_tax_lines_with_context(all_rec_texts_from_ocr, list(TAX_KEYWORDS))
        total_tax_final = sum(tax_amounts_detected)
        print(f"DEBUG process_data_custom: Calculated tax from global text: {total_tax_final}")

    final_result_json = {'status': 'success', 'tax': total_tax_final, 'items': valid_items_final }
    if not valid_items_final and total_tax_final == 0:
        final_result_json['status_detail'] = 'no_valid_items_or_tax_finalized'
    elif not valid_items_final:
        final_result_json['status_detail'] = 'no_valid_items_finalized'
    return final_result_json

def call_ocr_space_api(image_bytes: bytes, api_key: str, language: str = 'eng', 
                       ocr_engine: int = 2, is_table: bool = False, 
                       return_full_json: bool = False) -> Union[str, Dict, None]:
    payload = { 'apikey': api_key, 'language': language, 'isOverlayRequired': False, 'scale': True,
                'detectOrientation': True, 'OCREngine': ocr_engine, 'isTable': is_table }
    try:
        r = requests.post('https://api.ocr.space/parse/image', files={'filename': ('image.png', image_bytes)}, data=payload, timeout=45)
        r.raise_for_status(); result = r.json()
        if result.get('IsErroredOnProcessing'): return None if return_full_json else ""
        if not result.get('ParsedResults') or not result['ParsedResults'][0]: return None if return_full_json else ""
        parsed_text_output = result['ParsedResults'][0].get('ParsedText', "").strip()
        return result if return_full_json else parsed_text_output
    except Exception as e:
        print(f"ERROR in call_ocr_space_api: {e}")
        return None if return_full_json else ""

keywords_to_trigger_extraction = ['total', 'subtotal', 'tagihan', 'tunai', 'kembali', 'indomaret', 'alfamart', 
                                  'invoice', 'kepada', 'rp', 'penjaringan', 'purchase', 'pasta', 'kemang', 
                                  'table #', 'harga', 'jumlah', 'bayar', 'bill', 'receipt', 'faktur']

def read_image(image_np: np.ndarray, api_key: str):
    image_for_ocr_global_np = preprocess_for_ocr(image_np.copy())
    _, buffer_global = cv2.imencode('.png', image_for_ocr_global_np)
    image_bytes_global = buffer_global.tobytes()
    
    parsed_text_global_str = call_ocr_space_api(image_bytes_global, api_key, ocr_engine=2)
    if parsed_text_global_str is None:
        return {'status': 'error', 'message': 'Global OCR API call failed or returned no result.'}
    
    rec_texts_global_lines = [line.strip() for line in parsed_text_global_str.splitlines() if line.strip()]
    if not rec_texts_global_lines:
        return {'status': 'error', 'message': 'No text was detected by Global OCR.'}
    
    detected_global_lower = " ".join([t.lower() for t in rec_texts_global_lines if t])
    if not any(keyword in detected_global_lower for keyword in keywords_to_trigger_extraction):
        return {'status': 'error', 'message': 'Image does not appear to be a receipt/invoice (missing keywords).','ocr_text_global_for_reference': "\n".join(rec_texts_global_lines[:5])}

    preprocessed_yolo_img = preprocess(image_np.copy()) 
    boxes, classes = process_model(preprocessed_yolo_img)
    
    all_lines_from_table_crops_str = "" 
    if boxes is not None and classes is not None and 0 in classes:
        print("DEBUG read_image: Table detected by YOLO model.")
        temp_lines_from_crops = []
        for i_box, box_class_val in enumerate(classes):
            if box_class_val == 0: 
                bbox = boxes[i_box]
                h_orig,w_orig=image_np.shape[:2];h_yolo,w_yolo=preprocessed_yolo_img.shape[:2]
                x1o,y1o,x2o,y2o=int(bbox[0]*w_orig/w_yolo),int(bbox[1]*h_orig/h_yolo),int(bbox[2]*w_orig/w_yolo),int(bbox[3]*h_orig/h_yolo)
                cropped_img_np=crop_image_by_bbox(image_np.copy(),[x1o,y1o,x2o,y2o])
                if cropped_img_np is None or cropped_img_np.size==0: continue
                preprocessed_crop_ocr=preprocess_for_ocr(cropped_img_np)
                if preprocessed_crop_ocr.size==0: continue
                _,buffer_crop=cv2.imencode('.png',preprocessed_crop_ocr); image_bytes_crop=buffer_crop.tobytes()
                parsed_text_crop_str = call_ocr_space_api(image_bytes_crop, api_key, ocr_engine=2, is_table=True) 
                if parsed_text_crop_str:
                    temp_lines_from_crops.extend([l.strip() for l in parsed_text_crop_str.splitlines() if l.strip()])
        if temp_lines_from_crops:
            all_lines_from_table_crops_str = "\n".join(temp_lines_from_crops)
            print(f"DEBUG read_image: Text from table crops:\n{all_lines_from_table_crops_str}")
        else:
            print("DEBUG read_image: Table detected but no text from crops. Using global OCR.")
            all_lines_from_table_crops_str = parsed_text_global_str
    else:
        print("DEBUG read_image: No table detected. Using global OCR text.")
        all_lines_from_table_crops_str = parsed_text_global_str

    is_likely_invoice = "invoice" in detected_global_lower or \
                        "kepada yth" in detected_global_lower or "yth." in detected_global_lower or \
                        "no. faktur" in detected_global_lower or "faktur penjualan" in detected_global_lower or \
                        "faktur pajak" in detected_global_lower
    
    items_to_be_custom_processed = []
    tax_for_receipt_pre_calc = None 
    status_from_extraction = "pending"
    final_processing_as_receipt_flag = not is_likely_invoice

    if is_likely_invoice:
        print("DEBUG read_image: Terdeteksi sebagai INVOICE.")
        items_to_be_custom_processed = parse_invoice_lines_from_text(all_lines_from_table_crops_str.splitlines())
        if not items_to_be_custom_processed and rec_texts_global_lines != all_lines_from_table_crops_str.splitlines():
             items_to_be_custom_processed = parse_invoice_lines_from_text(rec_texts_global_lines)
        
    else: 
        print("DEBUG read_image: Terdeteksi sebagai STRUK.")
        if not all_lines_from_table_crops_str:
            print("DEBUG read_image: Struk - Tidak ada teks untuk diproses.")
            status_from_extraction = "error_no_text"
        else:
            extracted_receipt_data = extract_ocr_items_v2(all_lines_from_table_crops_str)
            print(f"DEBUG read_image: Hasil dari extract_ocr_items_v2 = {extracted_receipt_data}")
            items_to_be_custom_processed = extracted_receipt_data.get("items", [])
            status_from_extraction = extracted_receipt_data.get("status", "error_unknown")

    if not items_to_be_custom_processed and status_from_extraction not in ["success", "no_items_extracted"]:
        message = f"Extraction failed or yielded no items. Status: {status_from_extraction}"
        if is_likely_invoice and not items_to_be_custom_processed: message = "No items could be extracted from the invoice."
        print(f"DEBUG read_image: Tidak ada item untuk process_data_custom. Pesan: {message}")
    
    final_structured_result = process_data_custom(
        items_to_be_custom_processed,
        rec_texts_global_lines, 
        is_receipt_logic=final_processing_as_receipt_flag,
        pre_calculated_tax=tax_for_receipt_pre_calc
    )

    if (status_from_extraction == "no_items_extracted" or not items_to_be_custom_processed) and \
       not final_structured_result.get('items'):
        current_tax = final_structured_result.get('tax', 0)
        if current_tax > 0 :
            final_structured_result['status'] = 'warning'
            final_structured_result['message'] = 'No items were extracted, but tax information was found globally.'
        else:
            final_structured_result['status'] = 'error'
            final_structured_result['message'] = 'No items or tax information could be extracted.'
        if 'status_detail' in final_structured_result: del final_structured_result['status_detail']
    elif final_structured_result.get('status_detail') == 'no_valid_items_finalized':
        final_structured_result['status'] = 'warning'
        final_structured_result['message'] = 'Items were extracted, but none were finalized after cleaning/filtering.'
        if 'status_detail' in final_structured_result: del final_structured_result['status_detail']
    elif final_structured_result.get('status_detail') == 'no_valid_items_or_tax_finalized':
        final_structured_result['status'] = 'error'
        final_structured_result['message'] = 'No valid items or tax could be finalized after processing.'
        if 'status_detail' in final_structured_result: del final_structured_result['status_detail']
    elif not final_structured_result.get('items') and final_structured_result.get('tax', 0) == 0 and status_from_extraction not in ["success", "no_items_extracted"]:
        final_structured_result['status'] = 'error'
        final_structured_result['message'] = f'Failed to extract meaningful data. Extraction status: {status_from_extraction}'
    return final_structured_result

def crop_image_by_bbox(image: np.ndarray, bbox: list) -> Union[np.ndarray, None]:
    h, w = image.shape[:2];
    if len(bbox) == 4:
        x_min,y_min,x_max,y_max=map(int,bbox);
        x_min=max(0,min(x_min,w-1));x_max=max(0,min(x_max,w-1));
        y_min=max(0,min(y_min,h-1));y_max=max(0,min(y_max,h-1));
        if x_min>=x_max or y_min>=y_max: return None
        return image[y_min:y_max,x_min:x_max]
    return None