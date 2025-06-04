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

##DEFINE
label_map = {0: 'table', 1: 'not_table'}
RECEIPT_DISCOUNT_KEYWORDS = {"uc", "vc", "vc pt", "disc", "voucher", "diskon", "discount", "potongan"}

# --- FUNGSI UTILITAS PEMROSESAN ANGKA ---
def clean_digits_for_int(price_str_candidate: str) -> Union[int, None]:
    if not price_str_candidate: return None
    # MODIFIED LINE:
    cleaned_str = re.sub(r'^RP\s*\.?\s*', '', price_str_candidate, flags=re.IGNORECASE).strip()
    # END OF MODIFIED LINE
    cleaned_str = cleaned_str.replace(':', '.').replace('o', '0',-1).replace('O', '0',-1).replace('ộ', '0',-1)
    cleaned_str = re.sub(r'(\d)\s+(\d)', r'\1\2', cleaned_str)
    just_digits = re.sub(r'[.,]', '', cleaned_str)
    if just_digits.isdigit():
        try: return int(just_digits)
        except ValueError: return None
    return None

def extract_discount_info_from_line(line: str, discount_keywords: set) -> Union[Dict[str, Any], None]:
    """Mencari diskon pada baris. Mengembalikan dict {'text': desc, 'amount': val} atau None."""
    line_lower = line.lower()
    has_discount_keyword = any(keyword in line_lower for keyword in discount_keywords)
    
    # Pola untuk (angka) atau -angka atau keyword diikuti angka
    # Mengutamakan format dalam kurung atau dengan minus
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
                if amount and amount > 0 and amount < 50000: 
                     desc = line.replace(num_str, "").strip() or "Discount"
                     return {'text': desc.upper(), 'amount': amount, 'type': 'keyword_general_number'}
    return None


# --- FUNGSI PARSING STRUK ---
def clean_receipt_item_name_heuristic(name: str) -> str:
    if not name: return ""
    cleaned_name = name 
    match_qty_prefix = re.match(r'^(\d{1,2}\s?[Xx]?\s+)(.+)', name)
    if match_qty_prefix:
        cleaned_name = match_qty_prefix.group(2).strip() 
    else: 
         pass 

    cleaned_name = re.sub(r'[\s\#\+\*\@\:\-\/\(\)]+$', '', cleaned_name).strip()
    cleaned_name = re.sub(r'\s+(RP|Rp)$', '', cleaned_name, flags=re.IGNORECASE).strip()
    return cleaned_name.strip()


def parse_receipt_lines(
    receipt_lines: List[str],
    discount_keywords: set 
) -> List[Dict[str, Union[str, int]]]:
    parsed_items_gross = []
    PRICE_THRESHOLD = 0 
    i = 0
    
    while i < len(receipt_lines):
        current_line_original = receipt_lines[i].strip()
        i += 1 # Langsung naikkan i untuk current_line

        if not current_line_original: continue

        # Filter awal untuk baris yang kemungkinan besar bukan nama item (misal, hanya angka atau diskon)
        if re.fullmatch(r'\(\s*[\d,.]+\s*\)|-\s*[\d,.]+', current_line_original) or \
           (current_line_original.isdigit() and len(current_line_original) > 4 and i > len(receipt_lines) * 0.6): # Heuristik untuk angka di blok summary
            continue

        is_likely_summary_number_line = (
            re.fullmatch(r'[\d.,]+', current_line_original) and # Hanya angka, titik, koma
            len(re.sub(r'[.,]', '', current_line_original)) >= 1 and # Minimal satu digit angka
            i > len(receipt_lines) * 0.5 # Berada di paruh kedua struk (heuristik)
        )
        is_discount_format_line = re.fullmatch(r'\(\s*[\d,.]+\s*\)|-\s*[\d,.]+', current_line_original)

        if is_discount_format_line or is_likely_summary_number_line:
            continue

        item_name = ""
        item_price_gross = None
        
        match_item_price_same_line = re.match(r'^(.*?)(?:\s+|\s*([+-]?\s*(?:RP|Rp)?\s*[\d.,]{4,}))$', current_line_original)
        name_candidate_same_line = ""
        price_candidate_same_line_str = ""

        if match_item_price_same_line:
            potential_price_str = match_item_price_same_line.group(2)
            if potential_price_str:
                price_val_same_line = clean_digits_for_int(potential_price_str)
                if price_val_same_line is not None and price_val_same_line >= PRICE_THRESHOLD:
                    name_candidate_same_line = match_item_price_same_line.group(1).strip()
                    cleaned_name_cand = clean_receipt_item_name_heuristic(name_candidate_same_line)
                    
                    if len(cleaned_name_cand) >= 2 and \
                       not (cleaned_name_cand.isdigit() and len(cleaned_name_cand) > 3) and \
                       not old_is_noise(cleaned_name_cand) and \
                       not (re.fullmatch(r'\(\s*[\d,.]+\s*\)|-\s*[\d,.]+', cleaned_name_cand)):
                        item_name = cleaned_name_cand
                        item_price_gross = price_val_same_line

        if not item_name: 
            potential_item_name_curr_line = clean_receipt_item_name_heuristic(current_line_original)
            
            is_valid_potential_name = (
                len(potential_item_name_curr_line) >= 2 and
                not (potential_item_name_curr_line.isdigit() and len(potential_item_name_curr_line) > 3) and
                not old_is_noise(potential_item_name_curr_line) and
                not (re.fullmatch(r'\(\s*[\d,.]+\s*\)|-\s*[\d,.]+', potential_item_name_curr_line))
            )

            if is_valid_potential_name:
                if i < len(receipt_lines):
                    line1_after_name_str = receipt_lines[i].strip()
                    qty_match_single_digit = re.fullmatch(r'\d{1,3}', line1_after_name_str) # Kuantitas tunggal di satu baris

                    qty_price_match_on_line = re.match(r'^(\d{1,3})\s+([\d.,]{3,})$', line1_after_name_str) # Pola "QTY HARGA"

                    if qty_match_single_digit and (i + 1) < len(receipt_lines): # Kasus: NAMA -> QTY -> HARGA (masing2 di baris baru)
                        line2_after_name_str = receipt_lines[i+1].strip()
                        price_val_after_qty = clean_digits_for_int(line2_after_name_str)
                        if price_val_after_qty is not None and price_val_after_qty >= PRICE_THRESHOLD:
                            item_name = potential_item_name_curr_line
                            item_price_gross = price_val_after_qty
                            i += 2
                        
                    elif qty_price_match_on_line: 
                        price_str_from_qty_price = qty_price_match_on_line.group(2)
                        price_val = clean_digits_for_int(price_str_from_qty_price)
                        if price_val is not None and price_val >= PRICE_THRESHOLD:
                            item_name = potential_item_name_curr_line
                            item_price_gross = price_val
                        
                    elif not qty_match_single_digit:
                        price_val_direct = clean_digits_for_int(line1_after_name_str)
                        if price_val_direct is not None and price_val_direct >= PRICE_THRESHOLD:
                            item_name = potential_item_name_curr_line
                            item_price_gross = price_val_direct
                            i += 1
                            
        if item_name and item_price_gross is not None:
            #print(f"DEBUG PARSER: Menambahkan item='{item_name}', harga_kotor={item_price_gross}, dari_baris='{current_line_original}'") 
            parsed_items_gross.append({
                "item_name": item_name.upper(),
                "price": item_price_gross 
            })
            
    return parsed_items_gross

def extract_summary_values(receipt_lines: List[str]) -> List[Dict[str, int]]:
    summary_entries = []
    potential_summary_lines = []
    for line_idx in range(len(receipt_lines) - 1, -1, -1):
        line_stripped = receipt_lines[line_idx].strip()
        if re.fullmatch(r'[\d,.\s\(\)-]+', line_stripped) and any(char.isdigit() for char in line_stripped):
            potential_summary_lines.append(line_stripped)
        elif potential_summary_lines: 
            break 
        elif len(potential_summary_lines) == 0 and line_idx < len(receipt_lines) - 10:
            break
            
    potential_summary_lines.reverse() # Kembalikan ke urutan asli

    idx = 0
    while idx < len(potential_summary_lines):
        gross_price_str = potential_summary_lines[idx]
        # Pastikan ini bukan baris diskon dalam kurung yang berdiri sendiri sebagai harga kotor
        if re.fullmatch(r'\(\s*[\d,.]+\s*\)|-\s*[\d,.]+', gross_price_str):
            idx += 1
            continue

        gross_price = clean_digits_for_int(gross_price_str)
        if gross_price is None:
            idx += 1
            continue

        is_likely_qty_in_summary = (gross_price <= 10 and gross_price >= 0) 
        
        has_paired_discount = False
        if (idx + 1) < len(potential_summary_lines):
            next_line_str_check = potential_summary_lines[idx+1].strip()
            if re.fullmatch(r'\(\s*[\d,.]+\s*\)|-\s*[\d,.]+', next_line_str_check):
                has_paired_discount = True
                
        if is_likely_qty_in_summary and not has_paired_discount:
            #print(f"DEBUG SUMMARY_EXTRACT: Skipping likely QTY in summary: {gross_price_str}")
            idx += 1 
            continue

        discount_amount = 0
        if (idx + 1) < len(potential_summary_lines):
            next_line_str = potential_summary_lines[idx+1].strip()
            match_paren_discount = re.fullmatch(r'\(\s*([\d,.]+)\s*\)', next_line_str)
            match_neg_discount = re.fullmatch(r'-\s*([\d,.]+)', next_line_str)
            
            if match_paren_discount:
                discount_amount = clean_digits_for_int(match_paren_discount.group(1)) or 0
                summary_entries.append({'gross': gross_price, 'discount': discount_amount})
                idx += 2 # Konsumsi baris harga kotor dan baris diskon
                continue
            elif match_neg_discount:
                discount_amount = clean_digits_for_int(match_neg_discount.group(1)) or 0
                summary_entries.append({'gross': gross_price, 'discount': discount_amount})
                idx += 2 # Konsumsi baris harga kotor dan baris diskon
                continue
        
        # Jika tidak ada diskon berpasangan, harga kotor ini tidak ada diskonnya (dari summary)
        summary_entries.append({'gross': gross_price, 'discount': 0})
        idx += 1
            
    return summary_entries

def old_is_noise(text: str) -> bool:
    noise_keywords = ['subtotal', 'sub total', 'service', 'tax', 'pajak', 'pb1', 't0tal', 'subt0tal','r0unding','disk0n','disc0unt', 'vc', 'vc pt', 'rounding', 'diskon', 'discount', 'total', 'grand total', 'change', 'kembalian', 'srand tl', 'menu', 'price', 'jumlah', 'kembali']
    text_lower = text.lower()
    if text_lower == "vc": return False 
    return any(keyword in text_lower for keyword in noise_keywords)

def old_is_short_or_symbol(text: str, min_length=2) -> bool: 
    stripped = text.strip()
    if len(stripped) < min_length : return True
    if len(stripped) == 1 and not stripped.isalpha() and not stripped.isdigit(): return True
    return False

def extract_tax_values_from_line_by_keyword(line_text: str, tax_keywords: List[str]) -> List[int]:
    tax_amounts = []
    for keyword in tax_keywords:
        pattern = rf"(?:{re.escape(keyword)})\s*[:=\-]?\s*(?:(?:RP|Rp)\.?\s*)?([\d.,]+)"
        matches = re.findall(pattern, line_text, flags=re.IGNORECASE)
        for raw_amount in matches:
            if "%" in raw_amount: continue
            amount = clean_digits_for_int(raw_amount)
            if amount is not None and amount > 0 : tax_amounts.append(amount)
    return tax_amounts

def extract_tax_lines_with_context(rec_texts: List[str], tax_keywords: List[str] = None) -> Tuple[List[str], List[int]]:
    if tax_keywords is None: tax_keywords = ["ppn", "pajak", "tax", "pjk", "service", "serv", "pb1", "pb 1", "charge", "chrg"]
    all_tax_amounts = []; detected_tax_line_texts = []
    for i, line_text in enumerate(rec_texts):
        keyword_present = any(re.search(r'\b' + re.escape(kw) + r'\b', line_text, re.IGNORECASE) for kw in tax_keywords)
        if keyword_present:
            amounts_from_line = extract_tax_values_from_line_by_keyword(line_text, tax_keywords)
            if amounts_from_line:
                all_tax_amounts.extend(amounts_from_line); detected_tax_line_texts.append(line_text)
            elif i + 1 < len(rec_texts):
                next_line_text = rec_texts[i+1]
                price_in_next_line = clean_digits_for_int(next_line_text)
                if price_in_next_line and price_in_next_line > 0 and \
                   not any(re.search(r'\b' + re.escape(kw) + r'\b', next_line_text, re.IGNORECASE) for kw in tax_keywords):
                     if not any(noise_kw in line_text.lower() for noise_kw in ['subtotal', 'total', 'tagihan', 'bayar']):
                        all_tax_amounts.append(price_in_next_line); detected_tax_line_texts.append(f"{line_text} -> {next_line_text}")
    return list(set(detected_tax_line_texts)), all_tax_amounts

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
        elif len(cells) >= 2: raw_price_str = cells[1].get("Text", "").strip()
        price_val = clean_digits_for_int(raw_price_str)
        if item_name_str and price_val is not None and price_val > 0:
            lower_item_name = item_name_str.lower()
            if any(kw in lower_item_name for kw in ["keterangan", "description", "harga", "price", "jml", "qty", "total", "amount", "sub total", "pajak", "pembayaran", "kepada :", "no invoice :", "tanggal :"]):
                if len(item_name_str.split()) <= 2 and any(kw == lower_item_name for kw in ["keterangan", "harga", "jml", "total"]): continue
            parsed_items.append({"item_name": item_name_str.upper(), "price": price_val}) 
    return parsed_items

def parse_invoice_lines_from_text(invoice_text_lines: List[str]) -> List[Dict]:
    parsed_items = []
    for line_idx, line in enumerate(invoice_text_lines):
        original_line_for_debug = line; line_lower = line.lower()
        skip_keywords = ["kepada :", "tanggal :", "no invoice :", "pembayaran :", "sub total", "total rp", "pajak rp", "terimakasih atas", "fashion terlengkap", "salford & co."]
        if any(line_lower.startswith(kw) for kw in skip_keywords) or \
           line_lower in ["keterangan", "harga", "jml", "total"]: continue
        numbers_in_line = []
        for match in re.finditer(r'(?:RP\s*\.?\s*)?(\d[\d.,]*\d|\d+)', line, re.IGNORECASE):
            val = clean_digits_for_int(match.group(1));
            if val is not None: numbers_in_line.append({'value': val, 'start': match.start(), 'end': match.end()})
        if not numbers_in_line: continue
        item_price = None; price_start_index = -1
        for num_data in reversed(numbers_in_line):
            if num_data['value'] >= 1000: item_price = num_data['value']; price_start_index = num_data['start']; break 
        if item_price is None: continue
        first_significant_number_start_index = price_start_index
        if numbers_in_line: first_significant_number_start_index = numbers_in_line[0]['start']
        item_name = line[:first_significant_number_start_index].strip()
        item_name = re.sub(r'\s+(?:RP|Rp)\.?\s*$', '', item_name, flags=re.IGNORECASE).strip()
        final_item_name = clean_item_name(item_name) 
        if final_item_name and len(final_item_name) > 1 and not final_item_name.isdigit() and \
           final_item_name.lower() not in ["keterangan", "harga", "jml", "total", "rp"]:
            parsed_items.append({"item_name": final_item_name.upper(), "price": item_price})
    return parsed_items

def process_data_custom(
    parsed_items_list: List[Dict], 
    all_rec_texts_from_ocr: List[str],
    is_receipt_logic: bool = True
) -> Dict:
    valid_items_final = []
    for item_data in parsed_items_list: # Sudah berisi item dengan harga (mungkin sudah dikurangi diskon)
        item_name_upper = item_data.get('item_name', "").strip()
        price = item_data.get('price')

        if not item_name_upper or price is None : continue
        
        item_name_cleaned = clean_item_name(item_name_upper) 
        item_name_title_case = item_name_cleaned.title() 

        min_len = 2
        # Gunakan filter noise/short yang lebih sederhana di sini karena parsing utama sudah terjadi
        if len(item_name_title_case) < min_len : continue
        if item_name_title_case.isdigit() and not (item_name_title_case.lower() == "vc" and is_receipt_logic) : continue
        if old_is_noise(item_name_title_case) and not (item_name_title_case.lower() == "vc" and is_receipt_logic): continue
        
        if price < 0 : price = 0 # Harga tidak boleh negatif
        valid_items_final.append({'item_name': item_name_title_case, 'price': price})
    
    tax_keywords_list = ["ppn", "pajak", "tax", "pjk", "service", "serv", "pb1", "pb 1"]
    _, tax_amounts_detected = extract_tax_lines_with_context(all_rec_texts_from_ocr, tax_keywords_list)
    total_tax = sum(tax_amounts_detected)

    final_result_json = {'status': 'success', 'tax': total_tax, 'items': valid_items_final }
    if not valid_items_final :
        final_result_json['status_detail'] = 'no_valid_items_finalized'
    return final_result_json

# --- Fungsi API Call & `read_image` ---
def call_ocr_space_api(image_bytes: bytes, api_key: str, language: str = 'eng', 
                       ocr_engine: int = 2, is_table: bool = False, 
                       return_full_json: bool = False) -> Union[str, Dict, None]:
    payload = { 'apikey': api_key, 'language': language, 'isOverlayRequired': False, 'scale': True,
                'detectOrientation': True, 'OCREngine': ocr_engine, 'isTable': is_table }
    try:
        r = requests.post('https://api.ocr.space/parse/image', files={'filename': ('image.png', image_bytes)}, data=payload, timeout=45)
        r.raise_for_status(); result = r.json()
        print(f"DEBUG OCR Response (isTable={is_table}): {result}") # DEBUG
        if result.get('IsErroredOnProcessing'): return None if return_full_json else ""
        if not result.get('ParsedResults') or not result['ParsedResults'][0]: return None if return_full_json else ""
        return result if return_full_json else result['ParsedResults'][0].get('ParsedText', "").strip()
    except Exception as e:
        # print(f"ERROR in call_ocr_space_api: {e}") # DEBUG
        return None if return_full_json else ""

keywords_to_trigger_extraction = ['total', 'subtotal', 'tagihan', 'tunai', 'kembali', 'indomaret', 'alfamart', 'invoice', 'kepada', 'rp', 'penjaringan', 'purchase', 'pasta', 'kemang', 'table #']

def read_image(image_np: np.ndarray, api_key: str):
    image_for_ocr_global_np = preprocess_for_ocr(image_np.copy())
    _, buffer_global = cv2.imencode('.png', image_for_ocr_global_np)
    image_bytes_global = buffer_global.tobytes()
    parsed_text_global_str = call_ocr_space_api(image_bytes_global, api_key, ocr_engine=2)
    if parsed_text_global_str is None: return {'status': 'error', 'message': 'Global OCR API call failed. Please try again.'}
    rec_texts_global_lines = [line.strip() for line in parsed_text_global_str.splitlines() if line.strip()]
    if not rec_texts_global_lines: return {'status': 'error', 'message': 'No text was detected in the image by Global OCR.'}
    detected_global_lower = " ".join([t.lower() for t in rec_texts_global_lines if t])
    if not any(keyword in detected_global_lower for keyword in keywords_to_trigger_extraction):
        return {'status': 'error', 'message': 'The uploaded image does not appear to be a supported receipt or invoice, or essential keywords are missing.','ocr_text_global_for_reference': "\n".join(rec_texts_global_lines[:5])}

    preprocessed_yolo_img = preprocess(image_np.copy()) 
    boxes, classes = process_model(preprocessed_yolo_img)
    if boxes is None or classes is None: return {'status': 'error', 'message': 'No objects were detected by the layout model in the image.'}
    if 0 not in classes: return {'status': 'error', 'message': 'A valid itemization table (required for item extraction) was not detected by the layout model. The image may not be a supported type or is a new format not yet supported by Kakeibo.'}
    
    all_lines_from_table_crops = []
    items_from_invoice_json_table = []
    is_likely_invoice = "invoice" in detected_global_lower or "kepada :" in detected_global_lower or "no invoice" in detected_global_lower
    
    for i, box_class in enumerate(classes):
        if box_class == 0: 
            bbox = boxes[i]
            h_orig,w_orig=image_np.shape[:2];h_yolo,w_yolo=preprocessed_yolo_img.shape[:2]
            x1o,y1o,x2o,y2o=int(bbox[0]*w_orig/w_yolo),int(bbox[1]*h_orig/h_yolo),int(bbox[2]*w_orig/w_yolo),int(bbox[3]*h_orig/h_yolo)
            cropped_img_np=crop_image_by_bbox(image_np.copy(),[x1o,y1o,x2o,y2o])
            if cropped_img_np is None or cropped_img_np.size==0: continue
            preprocessed_crop_ocr=preprocess_for_ocr(cropped_img_np)
            if preprocessed_crop_ocr.size==0: continue
            _,buffer_crop=cv2.imencode('.png',preprocessed_crop_ocr); image_bytes_crop=buffer_crop.tobytes()

            if is_likely_invoice: 
                json_resp_table = call_ocr_space_api(image_bytes_crop, api_key, ocr_engine=1, is_table=True, return_full_json=True)
                if json_resp_table:
                    parsed_table_items = parse_ocr_space_table_for_invoice(json_resp_table)
                    if parsed_table_items: items_from_invoice_json_table.extend(parsed_table_items)
                    table_area_text = json_resp_table.get("ParsedResults",[{}])[0].get("ParsedText","")
                    if table_area_text: all_lines_from_table_crops.extend([l.strip() for l in table_area_text.splitlines() if l.strip()])
            else: 
                parsed_text_crop_str = call_ocr_space_api(image_bytes_crop, api_key, ocr_engine=2)
                if parsed_text_crop_str:
                    all_lines_from_table_crops.extend([l.strip() for l in parsed_text_crop_str.splitlines() if l.strip()])
    
    parsed_items_gross = parse_receipt_lines(all_lines_from_table_crops, RECEIPT_DISCOUNT_KEYWORDS)
    print(f"DEBUG READ_IMAGE: parsed_items_gross = {parsed_items_gross}") # DEBUG

    summary_details = extract_summary_values(all_lines_from_table_crops)
    print(f"DEBUG READ_IMAGE: summary_details = {summary_details}") # DEBUG

    # ---> PERBAIKAN KRUSIAL DI SINI <---
    final_items_to_process = [] # PASTIKAN DIINISIALISASI SEBAGAI LIST KOSONG!

    temp_parsed_items_gross = list(parsed_items_gross) # Buat salinan untuk diiterasi
    matched_summary_indices = [False] * len(summary_details)

    for item_g_idx, item_g in enumerate(temp_parsed_items_gross):
        item_applied_from_summary = False
        for summ_idx, summ_entry in enumerate(summary_details):
             print(f"DEBUG SUMMARY LOOP: Membandingkan dengan summ_entry = {summ_entry}") # DEBUG
             if not matched_summary_indices[summ_idx] and item_g['price'] == summ_entry['gross']:
                print(f"DEBUG SUMMARY LOOP: COCOK! Harga bersih = {summ_entry['gross'] - summ_entry['discount']}") # DEBUG
                final_items_to_process.append({
                    "item_name": item_g["item_name"],
                    "price": summ_entry['gross'] - summ_entry['discount'], # Harga bersih
                    # "original_price": summ_entry['gross'], # Opsional
                    # "discount_value": summ_entry['discount'] # Opsional
                })
                matched_summary_indices[summ_idx] = True
                item_applied_from_summary = True
                break 

        if not item_applied_from_summary:
            print(f"DEBUG SUMMARY LOOP: TIDAK COCOK untuk item_g = {item_g}, harga kotor dipertahankan.")
            final_items_to_process.append({
                "item_name": item_g["item_name"],
                "price": item_g["price"]
            })

    processing_as_receipt_flag = not is_likely_invoice

    if is_likely_invoice: 
        if items_from_invoice_json_table:
            final_items_to_process = items_from_invoice_json_table; processing_as_receipt_flag = False
        elif all_lines_from_table_crops: 
            invoice_items_fallback = parse_invoice_lines_from_text(all_lines_from_table_crops)
            if invoice_items_fallback: final_items_to_process = invoice_items_fallback; processing_as_receipt_flag = False
    
    if not final_items_to_process and all_lines_from_table_crops: 
        if not is_likely_invoice: processing_as_receipt_flag = True
        if processing_as_receipt_flag:
            final_items_to_process = parse_receipt_lines(all_lines_from_table_crops, RECEIPT_DISCOUNT_KEYWORDS)
    
    if not final_items_to_process:
        tax_only_result = process_data_custom([], rec_texts_global_lines, is_receipt_logic=processing_as_receipt_flag)
        return {'status': 'error', 'message': 'No items could be extracted from the detected table regions. Tax information (if any) is based on global OCR.','tax': tax_only_result.get('tax', 0),'items': []}

    final_structured_result = process_data_custom(final_items_to_process, rec_texts_global_lines, is_receipt_logic=processing_as_receipt_flag)
    
    if final_structured_result.get('status_detail') == 'no_valid_items_finalized':
        final_structured_result['status'] = 'warning'
        final_structured_result['message'] = 'Item processing of table regions was performed, but no valid items were finalized. This may be due to OCR quality or structure within the table regions.'
        del final_structured_result['status_detail']
    return final_structured_result

# --- Helper Functions ---
def clean_item_name(name: str) -> str:
    name = re.sub(r"^[^\w\d\s\-/()]+|[^\w\d\s\-/()]+$", "", name.strip())
    return name

def crop_image_by_bbox(image: np.ndarray, bbox: list) -> Union[np.ndarray, None]:
    h, w = image.shape[:2];
    if len(bbox) == 4:
        x_min,y_min,x_max,y_max=map(int,bbox);
        x_min=max(0,min(x_min,w-1));x_max=max(0,min(x_max,w-1));
        y_min=max(0,min(y_min,h-1));y_max=max(0,min(y_max,h-1));
        if x_min>=x_max or y_min>=y_max: return None
        return image[y_min:y_max,x_min:x_max]
    return None