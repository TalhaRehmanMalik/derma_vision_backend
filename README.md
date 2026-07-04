# Derma Vision — FastAPI Backend

AI-powered skin cancer classification system built with FastAPI, MobileNetV2, and Groq Llama-3.

---

## Project Structure

```
derma_vision_backend/
│
├── main.py                        ← FastAPI app entry point
├── database.py                    ← MySQL connection + auto-create database
├── models.py                      ← SQLAlchemy ORM models (Users, Scans)
│
├── routes/
│   ├── __init__.py
│   ├── auth.py                    ← Register, OTP verify, Login, Password reset
│   ├── predict.py                 ← Image upload + ML inference + scan history
│   ├── chat.py                    ← Groq Llama-3 dermatology chatbot
│   └── admin.py                   ← Admin stats, user management, scan management
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                  ← Central logger (used across all modules)
│   ├── auth.py                    ← JWT create/verify + admin guard
│   ├── email.py                   ← Gmail SMTP OTP sender
│   └── inference.py               ← MobileNetV2 model load + predict
│
├── ml_models/
│   ├── derma_vision_mobilenetv2.h5    ← Trained Keras model
│   └── class_mapping.json             ← Class index to name mapping
│
├── uploads/                       ← Auto-created — stores scan images
├── venv/                          ← Python virtual environment
├── .env                           ← Environment config (never commit this)
├── requirements.txt
└── README.md
```

---

## Skin Conditions Classified

| Code  | Condition              | Description                        |
|-------|------------------------|------------------------------------|
| MEL   | Melanoma               | Most dangerous skin cancer         |
| NV    | Melanocytic Nevi       | Common moles, usually benign       |
| BCC   | Basal Cell Carcinoma   | Most common skin cancer            |
| AKIEC | Actinic Keratosis      | Precancerous skin condition        |

---

## Setup — Step by Step

### Step 1 — Clone / Extract Project

```bash
cd derma_vision_backend
```

### Step 2 — Create Python Virtual Environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Place ML Model Files

Put these 2 files inside the `ml_models/` folder:
- `derma_vision_mobilenetv2.h5`
- `class_mapping.json`

### Step 5 — Configure .env File

Create a `.env` file in the root folder:

```env
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=derma_vision

# JWT
JWT_SECRET=your_strong_secret_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24

# Email — Gmail SMTP (use App Password, not your login password)
# Get App Password: https://myaccount.google.com/apppasswords
MAIL_USER=your_email@gmail.com
MAIL_PASS=your_gmail_app_password

# Groq API — free key from https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# File storage
UPLOAD_FOLDER=uploads

# Logging — INFO for production, DEBUG for development
LOG_LEVEL=INFO
```

### Step 6 — Start Server

```bash
uvicorn main:app --reload --port 8000
```

On startup the server automatically:
- Creates the `derma_vision` MySQL database if it does not exist
- Creates `users` and `scans` tables
- Loads the MobileNetV2 model into memory

### Step 7 — Test the API

Open in browser: [http://localhost:8000/docs](http://localhost:8000/docs)

Swagger UI will open — all endpoints can be tested there.

---

## API Endpoints

### Auth — `/api/auth`

| Method | Endpoint                    | Auth | Description                          |
|--------|-----------------------------|------|--------------------------------------|
| POST   | `/api/auth/register`        | ✗    | Register new account + send OTP email |
| POST   | `/api/auth/verify-otp`      | ✗    | Verify OTP → returns JWT token        |
| POST   | `/api/auth/login`           | ✗    | Login with email + password → JWT     |
| POST   | `/api/auth/forgot-password` | ✗    | Send new OTP to email                 |
| POST   | `/api/auth/reset-password`  | ✗    | Reset password using OTP              |

### Predict — `/api`

| Method | Endpoint              | Auth | Description                            |
|--------|-----------------------|------|----------------------------------------|
| POST   | `/api/predict`        | JWT  | Upload skin image → AI diagnosis       |
| GET    | `/api/scans`          | JWT  | Get current user's scan history        |
| DELETE | `/api/scans/{id}`     | JWT  | Delete own scan by ID                  |

### Chat — `/api`

| Method | Endpoint     | Auth | Description                              |
|--------|--------------|------|------------------------------------------|
| POST   | `/api/chat`  | JWT  | Ask dermatology question to Groq Llama-3 |

### Admin — `/api/admin`

| Method | Endpoint                    | Auth  | Description                          |
|--------|-----------------------------|-------|--------------------------------------|
| GET    | `/api/admin/stats`          | Admin | System stats + class distribution    |
| GET    | `/api/admin/users`          | Admin | All users with scan counts           |
| GET    | `/api/admin/scans`          | Admin | All scans paginated                  |
| DELETE | `/api/admin/scans/{id}`     | Admin | Delete any scan + its image file     |
| DELETE | `/api/admin/users/{id}`     | Admin | Delete user + all their data         |

### Helper

| Method | Endpoint                    | Auth  | Description                          |
|--------|-----------------------------|-------|--------------------------------------|
| POST   | `/make-admin/{user_id}`     | Admin | Promote a user to admin              |
| GET    | `/`                         | ✗     | Health check                         |

---

## Authentication Flow

```
Register → OTP sent to email → Verify OTP → JWT returned → Use JWT in headers
```

All protected endpoints require:
```
Authorization: Bearer <your_jwt_token>
```

---

## First Admin Setup

Since `/make-admin` requires an existing admin to call it, the very first admin
must be set directly in the database. Do this once after registering:

```sql
USE derma_vision;
UPDATE users SET is_admin=1 WHERE id=1;
```

After that, use the `/make-admin/{user_id}` endpoint for all future promotions.

---

## Logging

All modules use the central logger from `utils/logger.py`.
Logs are printed to stdout in this format:

```
2026-06-25 09:42:00 | INFO     | main          | All tables created/verified.
2026-06-25 09:42:01 | INFO     | inference     | Model loaded. Classes: [...]
2026-06-25 09:43:10 | INFO     | routes.auth   | New user registered: ali (ali@gmail.com)
2026-06-25 09:43:15 | WARNING  | auth          | Invalid OTP attempt for ali@gmail.com
2026-06-25 09:43:20 | ERROR    | routes.chat   | Groq API timed out
```

Control log level in `.env`:
```env
LOG_LEVEL=INFO    # production
LOG_LEVEL=DEBUG   # development
```

---

## ML Model — MobileNetV2

- Input: 224 × 224 RGB image
- Output: Probability scores for 4 skin condition classes
- Confidence threshold: 60% — below this result is marked inconclusive
- Loaded once at startup, reused for every request

---

## Security Notes

- Passwords are hashed with `bcrypt` — never stored as plain text
- JWT tokens expire after 24 hours (configurable via `JWT_EXPIRE_HOURS`)
- OTP codes expire after 10 minutes
- Uploaded images are saved with UUID filenames — prevents path traversal attacks
- `/make-admin` requires an existing admin JWT — not publicly accessible
- Forgot password response never reveals whether email exists in database

---

## Environment Variables Reference

| Variable          | Required | Default        | Description                        |
|-------------------|----------|----------------|------------------------------------|
| `DB_HOST`         | No       | `localhost`    | MySQL host                         |
| `DB_PORT`         | No       | `3306`         | MySQL port                         |
| `DB_USER`         | No       | `root`         | MySQL username                     |
| `DB_PASSWORD`     | Yes      | —              | MySQL password                     |
| `DB_NAME`         | No       | `derma_vision` | Database name                      |
| `JWT_SECRET`      | Yes      | —              | Secret key for signing JWT tokens  |
| `JWT_ALGORITHM`   | No       | `HS256`        | JWT signing algorithm              |
| `JWT_EXPIRE_HOURS`| No       | `24`           | Token expiry in hours              |
| `MAIL_USER`       | No       | —              | Gmail address for OTP emails       |
| `MAIL_PASS`       | No       | —              | Gmail App Password                 |
| `GROQ_API_KEY`    | No       | —              | Groq API key for chatbot           |
| `UPLOAD_FOLDER`   | No       | `uploads`      | Folder to store scan images        |
| `LOG_LEVEL`       | No       | `INFO`         | Logging level (INFO/DEBUG/ERROR)   |

---

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Framework   | FastAPI                             |
| Database    | MySQL + SQLAlchemy ORM              |
| ML Model    | TensorFlow / Keras — MobileNetV2   |
| Auth        | JWT (python-jose) + bcrypt          |
| Chatbot     | Groq Llama-3 (llama3-8b-8192)      |
| Email       | Gmail SMTP                          |
| Logging     | Python logging module               |