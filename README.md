# Fly Counting System

Fly Counting System adalah sistem deteksi dan perhitungan lalat menggunakan Computer Vision dengan model YOLOv5 custom training yang diakses melalui FastAPI ML Service dan diintegrasikan dengan Backend API Node.js.

Sistem ini digunakan untuk membantu teknisi lapangan melakukan monitoring populasi lalat dari fly catcher secara digital.

==================================================

SYSTEM OVERVIEW

Project ini terdiri dari 3 komponen utama:

1. Backend API
Backend digunakan untuk mengelola request dari client dan menyimpan data monitoring.

Technology:
- Node.js
- Express.js
- PostgreSQL


2. ML Detection Service

Service khusus untuk menjalankan model deteksi lalat menggunakan YOLOv5.

Technology:
- Python
- FastAPI
- YOLOv5
- PyTorch
- OpenCV

ML service menerima gambar fly catcher lalu mengembalikan:
- jumlah lalat
- jumlah agas
- bounding box deteksi


3. Database

Digunakan untuk menyimpan data monitoring hasil inspeksi teknisi.

Technology:
- PostgreSQL


==================================================

SYSTEM ARCHITECTURE

Client (Mobile / Web)
        |
        v
Backend API (Node.js)
        |
        v
ML Service (FastAPI + YOLOv5)
        |
        v
PostgreSQL Database


==================================================

PROJECT STRUCTURE

FlyCountingProject/

backend/
    server.js
    package.json
    package-lock.json
    prisma/
        schema.prisma

ml-service/
    main.py
    best.pt
    requirements.txt

README.md


==================================================

BACKEND SETUP

1. Masuk ke folder backend

cd backend

2. Install dependencies

npm install

3. Jalankan server

node server.js

Backend akan berjalan di:

http://localhost:5000


==================================================

ML SERVICE SETUP

1. Masuk ke folder ML Service

cd ml-service

2. Install Python dependencies

pip install -r requirements.txt

3. Jalankan ML API

uvicorn main:app --reload

ML service akan berjalan di:

http://localhost:8000


==================================================

HEALTH CHECK ENDPOINT

GET /

Response:

{
  "status": "ML Fly Detection API is running"
}


==================================================

PREDICTION ENDPOINT

POST /predict

Request:

form-data
key: file
value: image file


Response Example:

{
  "total": 12,
  "total_agas": 5,
  "total_lalat_hijau": 7
}


==================================================

DATABASE SETUP

Project menggunakan PostgreSQL.

1. Install PostgreSQL

Download dari:
https://www.postgresql.org/download/


2. Buat database baru

Contoh:

flydb


3. Update file .env pada backend

Contoh konfigurasi:

DATABASE_URL=postgresql://username:password@localhost:5432/flydb


==================================================

SYSTEM REQUIREMENTS

Minimum environment:

Python 3.9+
Node.js 18+
PostgreSQL 14+


==================================================

PYTHON DEPENDENCIES

Dependency utama ML service:

- fastapi
- uvicorn
- torch
- opencv-python
- pillow
- numpy
- pandas

Semua dependency tersedia di:

ml-service/requirements.txt


==================================================

DEPLOYMENT NOTES

Untuk deployment production disarankan menggunakan:

- VPS / Cloud Server
- Docker
- Reverse Proxy (Nginx)

File penting untuk ML service:

ml-service/best.pt

File ini adalah model hasil training YOLOv5 yang digunakan untuk deteksi lalat.


==================================================

SECURITY NOTES

- Jangan commit file .env
- Gunakan environment variable untuk credential database
- Batasi akses model file untuk keamanan


==================================================

PROJECT MAINTAINER

R&D Computer Vision Project
Fly Detection and Monitoring System
YOLOv5 + FastAPI Implementation


==================================================

LICENSE

Private Repository – Internal Use Only