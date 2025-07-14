# Receipt & Invoice Extraction API 🚀

API ini mengimplementasikan model *object detection* **YOLOv8** untuk mengenali tabel pada gambar struk (*receipt*) dan *invoice*. Setelah tabel terdeteksi, gambar akan dipotong dan dikirim ke layanan OCR.space untuk ekstraksi teks. Teks yang dihasilkan kemudian diproses untuk mengekstrak informasi penting seperti nama item, harga, dan pajak, lalu dikembalikan dalam format JSON.

API ini dibuat menggunakan **Flask** dan berjalan di **Python 3.11**.

-----

## ✨ Fitur Utama

  - **Deteksi Tabel**: Deteksi otomatis lokasi tabel pada gambar struk dan invoice menggunakan model **YOLOv8**.
  - **Pemotongan Cerdas**: Gambar akan dipotong (*crop*) secara presisi berdasarkan hasil deteksi untuk meningkatkan akurasi OCR.
  - **Integrasi OCR**: Terhubung dengan [OCR.space API](https://ocr.space/) untuk mengubah data gambar menjadi teks.
  - **Klasifikasi Dokumen**: Secara otomatis membedakan antara struk dan invoice berdasarkan kata kunci yang ditemukan.
  - **Ekstraksi Informasi**: Mengekstrak data penting seperti:
      - Nama Item/Produk
      - Harga per Item
      - Total Harga
      - Pajak (Tax)
  - **Output JSON**: Mengembalikan hasil ekstraksi dalam format JSON yang terstruktur dan mudah digunakan.

-----

## ⚙️ Alur Kerja API

1.  **Request Masuk**: Client mengirim request `POST` ke endpoint `/extract` dengan *body* `multipart/form-data` yang berisi file gambar.
2.  **Deteksi Objek**: Model **YOLOv8** dijalankan untuk menemukan *bounding box* dari tabel pada gambar.
3.  **Cropping**: Gambar dipotong sesuai dengan *bounding box* yang dideteksi.
4.  **Proses OCR**: Gambar yang sudah dipotong dikirim ke OCR.space API.
5.  **Klasifikasi & Ekstraksi**:
      - Hasil teks dari OCR dianalisis untuk menemukan kata kunci seperti "INVOICE" atau "RECEIPT".
      - Berdasarkan klasifikasi, fungsi ekstraksi yang sesuai (untuk invoice atau struk) akan dijalankan.
      - Proses ini melibatkan pembersihan teks (*cleaning*), pencarian harga, nama item, dan pajak.
6.  **Response JSON**: API mengembalikan data yang telah diekstraksi dalam format JSON.

-----

## 📦 Teknologi yang Digunakan

  - **Backend**: Flask
  - **Bahasa**: **Python 3.11**
  - **Object Detection**: **YOLOv8 (Ultralytics)**
  - **Layanan Eksternal**: OCR.space API

-----

# Receipt & Invoice Extraction API 🚀

This API implements a **YOLOv8** object detection model to recognize tables in receipt and invoice images. After a table is detected, the image is cropped and sent to the OCR.space service for text extraction. The resulting text is then processed to extract key information such as item names, prices, and tax, which is returned in JSON format.

This API is built using **Flask** and runs on **Python 3.11**.

---

## ✨ Key Features

-   **Table Detection**: Automatically detects the location of tables in receipt and invoice images using the **YOLOv8** model.
-   **Smart Cropping**: Precisely crops the image based on the detection results to improve OCR accuracy.
-   **OCR Integration**: Connects with the [OCR.space API](https://ocr.space/) to convert image data into text.
-   **Document Classification**: Automatically distinguishes between receipts and invoices based on found keywords.
-   **Information Extraction**: Extracts key data such as:
    -   Item/Product Name
    -   Price per Item
    -   Total Price
    -   Tax
-   **JSON Output**: Returns the extracted results in a structured and easy-to-use JSON format.

---

## ⚙️ API Workflow

1.  **Incoming Request**: A client sends a `POST` request to the `/extract` endpoint with a `multipart/form-data` body containing an image file.
2.  **Object Detection**: The **YOLOv8** model is run to find the bounding box of the table in the image.
3.  **Cropping**: The image is cropped according to the detected bounding box.
4.  **OCR Processing**: The cropped image is sent to the OCR.space API.
5.  **Classification & Extraction**:
    -   The text result from OCR is analyzed to find keywords like "INVOICE" or "RECEIPT".
    -   Based on the classification, the appropriate extraction function (for an invoice or receipt) is executed.
    -   This process involves text cleaning and finding prices, item names, and tax.
6.  **JSON Response**: The API returns the extracted data in JSON format.

---

## 📦 Technology Stack

-   **Backend**: Flask
-   **Language**: **Python 3.11**
-   **Object Detection**: **YOLOv8 (Ultralytics)**
-   **External Service**: OCR.space API
