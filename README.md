---
title: Derma Vision Backend
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Derma Vision Backend

AI-powered skin cancer classification system built with FastAPI and MobileNetV2.

## Project Structure

```
derma_vision_backend/
│
├── main.py                        ← FastAPI app entry point
├── database.py                    ← MySQL connection + auto-create database
├── models.py                      ← SQLAlchemy ORM models (Users, Scans)
│
├── routes/
│   ├── auth.py                    ← Register, OTP verify, Login, Password reset
│   ├── predict.py                 ← Image upload + ML inference + scan history
│   ├── chat.py                    ← Groq Llama-3 dermatology chatbot
│   └── admin.py                   ← Admin stats, user management
│
├── utils/
│   ├── logger.py                  ← Central logger
│   ├── auth.py                    ← JWT create/verify + admin guard
│   ├── email.py                   ← Brevo API OTP sender
│   └── inference.py               ← MobileNetV2 model load + predict
│
├── ml_models/
│   ├── derma_vision_mobilenetv2.h5
│   └── class_mapping.json
│
├── uploads/                       ← Auto-created scan images folder
├── .env                           ← Environment config
└── requirements.txt
```

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/register | Register + OTP email |
| POST | /api/auth/verify-otp | Verify OTP → JWT |
| POST | /api/auth/login | Login → JWT |
| POST | /api/auth/forgot-password | Send new OTP |
| POST | /api/auth/reset-password | Reset password |

### Predict (JWT required)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/predict | Upload image → diagnosis |
| GET | /api/scans | My scan history |
| DELETE | /api/scans/{id} | Delete scan |

### Chat (JWT required)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/chat | Dermatology Q&A |

### Admin (Admin JWT required)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/admin/stats | System statistics |
| GET | /api/admin/users | All users |
| GET | /api/admin/scans | All scans |
| DELETE | /api/admin/scans/{id} | Delete any scan |
| DELETE | /api/admin/users/{id} | Delete user + data |

## ML Model

- Architecture: MobileNetV2 (Transfer Learning from ImageNet)
- Dataset: HAM10000 — 4 classes (MEL, NV, BCC, AKIEC)
- Accuracy: 79.6% | AUC: 0.9556
- Training: Two-phase fine-tuning on Google Colab T4 GPU

## Tech Stack

- FastAPI + Uvicorn
- SQLAlchemy + MySQL (Clever Cloud)
- TensorFlow / Keras
- JWT Authentication + Bcrypt
- Groq Llama-3 Chatbot

## OTP email configuration

Verify a sender email in Brevo, then set these variables in `.env` or Hugging Face Secrets:

```env
BREVO_API_KEY=your_brevo_api_key
BREVO_SENDER_EMAIL=your_verified_sender@example.com
BREVO_SENDER_NAME=Derma Vision
```

The Hugging Face API token is separate from Brevo and should use the variable name expected by the Hugging Face integration.