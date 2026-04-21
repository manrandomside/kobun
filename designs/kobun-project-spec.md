# PROJECT SPECIFICATION DOCUMENT
# Kobun — Kuzushiji Document Recognition & Translation with Deep Learning

> **Versi**: 2.0 (Final — Full-Stack Architecture)  
> **Tanggal**: 20 April 2026  
> **Status**: Ready for Implementation  
> **Nama Project**: Kobun (古文 = Tulisan Klasik)  
> **Tagline**: "AI untuk membaca dokumen Jepang kuno yang hanya bisa dibaca 0,01% orang Jepang modern"  
> **Companion Project**: Kioku (記憶) — platform belajar bahasa Jepang (project terpisah, arsitektur independen)

---

## DAFTAR ISI

- [A. Konsep dan Tujuan Project](#a-konsep-dan-tujuan-project)
- [B. Tech Stack](#b-tech-stack)
- [C. Arsitektur Sistem & Repository](#c-arsitektur-sistem--repository)
- [D. Dataset & Data Pipeline](#d-dataset--data-pipeline)
- [E. Model Architecture & Training Plan](#e-model-architecture--training-plan)
- [F. Evaluation Methodology](#f-evaluation-methodology)
- [G. Frontend Design & UI Specification](#g-frontend-design--ui-specification)
- [H. Backend API & Integration](#h-backend-api--integration)
- [I. ML Server (HuggingFace Space)](#i-ml-server-huggingface-space)
- [J. Technical Report & Reproducibility](#j-technical-report--reproducibility)
- [K. Project Phases & Deliverables](#k-project-phases--deliverables)
- [L. Aturan Coding](#l-aturan-coding)

---

# A. KONSEP DAN TUJUAN PROJECT

## Nama Project

**Kobun** (古文) — berarti "Tulisan Klasik" atau "Literatur Klasik" dalam bahasa Jepang. Nama ini dipilih karena:

- **Makna yang tepat**: Langsung merujuk pada domain project — dokumen klasik Jepang yang ditulis dalam kuzushiji
- **Pararel dengan Kioku**: Sama-sama satu kata, 5 huruf, bermakna dalam bahasa Jepang
- **Mudah dieja**: `k-o-b-u-n`, international-friendly
- **Unik**: Belum ada tool ML populer yang menggunakan nama ini
- **Brand narrative**: Kioku (記憶/memori) + Kobun (古文/tulisan klasik) + [project Jepang berikutnya] = portfolio ecosystem yang koheren untuk karir di Jepang

## Arsitektur Project — FULL-STACK STANDALONE

**Kobun adalah project ML + fullstack yang berdiri sendiri, dengan arsitektur 3-tier.**

Arsitektur tinggi-tingkat:

```
User browser
    |
    v
Frontend (Next.js 15 + React 19 + shadcn/ui)
  Deploy: Vercel (free)
    |
    v HTTP API call
    |
Backend (FastAPI Python Serverless Functions)
  Deploy: Vercel Serverless Functions
  Role: Lightweight proxy + rate limiting
    |
    v HTTP API call
    |
ML Server (HuggingFace Space + Gradio)
  Deploy: HuggingFace Spaces (free CPU)
  Role: Public ML demo + inference API endpoint
  ML Pipeline: PyTorch + ONNX + Gemini
```

### Rasional Arsitektur 3-Tier

1. **Frontend custom (Next.js)** — Agar UI berkualitas sama dengan Kioku (shadcn/ui + Framer Motion), memberikan pengalaman premium ke recruiter
2. **Backend proxy (FastAPI)** — Untuk rate limiting + kontrol akses + hiding internal ML server details
3. **ML Server (HuggingFace Space)** — Tetap sebagai **public ML demo + API endpoint**, sehingga:
   - User yang mau quick-try bisa langsung akses HF Space tanpa buka website custom
   - Model cards, papers, dan blog post bisa mengutip URL HF Space secara kredibel
   - Full reproducibility untuk recruiter yang mau evaluate models langsung

### Keuntungan Arsitektur Ini

- **Dual deliverable**: Custom premium UI (Vercel) + Public ML demo (HF Space) — portfolio value maksimal
- **Separation of concerns**: UI, business logic, dan ML terpisah clean
- **$0 cost**: Semua tier menggunakan free tier
- **Zero-storage maintained**: Tidak ada database, tidak ada user accounts, tidak ada history
- **Scalable pattern**: Kalau nanti mau upgrade ke database atau auth, struktur sudah siap

### Prinsip Tetap Ada

- **Tidak ada database user** — Kobun sepenuhnya stateless
- **Tidak ada autentikasi** — anyone bisa akses demo
- **Tidak ada history/storage foto user** — zero-storage pattern
- **Public model + public demo + public code** — semua shareable

## Deskripsi Project

**Kobun** adalah sistem AI end-to-end berbasis deep learning yang mampu membaca, mengenali, dan mentranskripsikan karakter *kuzushiji* (くずし字) — gaya tulisan kursif Jepang kuno yang digunakan selama lebih dari 1000 tahun (abad ke-8 hingga akhir abad ke-19) namun hari ini hanya bisa dibaca oleh kurang dari 0,01% penduduk Jepang.

Sistem ini terdiri dari 3 komponen ML utama:

1. **Character Classifier** — CNN (ResNet-50) dan Vision Transformer yang mengklasifikasikan 49 karakter hiragana klasik (Kuzushiji-49 dataset)
2. **Character Detector** — Object detection model (YOLOv8/11) yang menemukan lokasi dan bounding box setiap karakter pada scan halaman dokumen kuno
3. **End-to-end Pipeline** — Menggabungkan deteksi + klasifikasi + reading order + mapping ke hiragana modern + terjemahan trilingual (Japanese, English, Indonesian)

**Output utama adalah 2 demo publik:**

1. **Custom Premium Web App** (Vercel) — Next.js frontend dengan shadcn/ui, Framer Motion animations, trilingual UI (JP default/EN/ID), sample gallery, drag-drop upload, beautiful results visualization
2. **Public ML Demo** (HuggingFace Spaces) — Gradio-based public demo yang bisa dicite di paper, model cards, dan blog post

**Output pendukung:**
- Trained models di HuggingFace Hub (public)
- Paper-style technical report (markdown + PDF)
- Blog post menjelaskan journey + technical decisions
- Reproducible training pipeline (config-driven, one-command reproduce)
- Demo video (60-90 detik)

## Tujuan Utama

1. **Portofolio Full-stack AI Engineering**: Mendemonstrasikan kemampuan full ML lifecycle (training → evaluation → deployment) + kemampuan fullstack development (modern frontend, API design, deployment orchestration). Kombinasi rare yang sangat dicari di 2026-2029
2. **Japan Cultural Impact**: Berkontribusi pada preservasi warisan budaya Jepang. <3 juta buku dan >1 miliar dokumen historis di Jepang masih belum ditranskripsikan. Kurang dari 0,01% penduduk Jepang bisa membaca kuzushiji
3. **Karir Signal untuk Jepang**: Posisi diri sebagai **Full-stack AI Engineer** yang punya narasi kuat — "Saya build sistem ML production-grade dengan UI premium, bukan hanya jupyter notebook"
4. **International Accessibility**: Demo trilingual (Japanese default, English, Indonesian) membuat project accessible untuk recruiter Jepang (primary audience), international audience, dan local Indonesian audience
5. **Complement Kioku**: Kobun + Kioku = 2 project dengan **quality floor yang sama** (Next.js + shadcn/ui + Tailwind), memperlihatkan konsistensi engineering standard
6. **Zero Cost**: Seluruh project berjalan pada free tier (Kaggle GPU, HuggingFace Spaces, Vercel Hobby, Gemini API free tier)

## Fitur Utama (Scope)

### Fase 1: Character Classification (Foundation)

| No | Deliverable | Deskripsi |
|----|-------------|-----------|
| 1 | **Dataset Analysis Notebook** | EDA Kuzushiji-49: class distribution, sample visualization, imbalance analysis |
| 2 | **Baseline Model** | Simple CNN, ResNet-18, benchmark accuracy |
| 3 | **Advanced Models** | ResNet-50 dengan manifold mixup, Vision Transformer (ViT) |
| 4 | **Training Pipeline** | Reproducible training script dengan config file, checkpointing, logging |
| 5 | **Evaluation Report** | Balanced accuracy, per-class accuracy, confusion matrix, error analysis |
| 6 | **Model Registry** | Semua trained models upload ke HuggingFace Hub dengan model cards |

### Fase 2: Full-Page Character Detection + ML Pipeline

| No | Deliverable | Deskripsi |
|----|-------------|-----------|
| 7 | **Kaggle Kuzushiji Recognition Dataset** | Download + preprocess (halaman penuh dengan bounding box annotations) |
| 8 | **YOLO Detection Model** | Fine-tune YOLOv8/YOLO11 untuk deteksi karakter kuzushiji |
| 9 | **Detection Evaluation** | mAP, Precision/Recall, F1 score pada test set |
| 10 | **Two-Stage Pipeline** | Detect → Crop → Classify → Sort by reading order |
| 11 | **Reading Order Logic** | Teks Jepang klasik dibaca top-to-bottom, right-to-left |
| 12 | **Trilingual Translation Layer** | Kuzushiji unicode → Modern kana → Japanese modern / English / Indonesian (Gemini API) |

### Fase 3: ML Server (HuggingFace Space)

| No | Deliverable | Deskripsi |
|----|-------------|-----------|
| 13 | **Gradio App (Public ML Demo)** | User upload image → preview → detection → classification → translation |
| 14 | **ONNX Optimization** | Model di-convert ke ONNX + quantize untuk CPU inference <3 detik |
| 15 | **Sample Gallery** | 10-15 contoh dokumen historis public domain pre-loaded |
| 16 | **Auto-Generated API** | Gradio auto-exposed REST API endpoint untuk dipanggil dari frontend custom |
| 17 | **Trilingual UI (Gradio)** | Language switcher Japanese (default) / English / Indonesian |
| 18 | **Zero-Storage Architecture** | Image processed in-memory, tidak pernah disimpan |
| 19 | **Public URL** | `https://[username]-kobun.hf.space` |
| 20 | **Bilingual README** | English + Japanese README di Space |

### Fase 4: Custom Frontend + Backend API

| No | Deliverable | Deskripsi |
|----|-------------|-----------|
| 21 | **Next.js 15 Frontend** | App Router + React 19 + TypeScript 5 |
| 22 | **shadcn/ui Components** | Polished UI dengan Tailwind CSS v4 |
| 23 | **Framer Motion Animations** | Page transitions, upload drag-drop, result reveal |
| 24 | **Trilingual UI (Next.js)** | next-intl untuk JP (default) / EN / ID |
| 25 | **Hero Landing Page** | Visual impressive, hero image + CTA + sample showcase |
| 26 | **Upload Flow** | Drag-drop zone, file validation, progress indicator |
| 27 | **Results Viewer** | Annotated image + transcription + trilingual translation |
| 28 | **FastAPI Backend** | Python serverless functions di Vercel, proxy ke HF Space |
| 29 | **Rate Limiting** | In-memory sliding window (10 requests/menit per IP) |
| 30 | **Error Handling** | Graceful fallbacks, informative error messages |
| 31 | **SEO & Metadata** | Meta tags, Open Graph, structured data |
| 32 | **Responsive Design** | Mobile-first, tablet, desktop |
| 33 | **Dark/Light Mode** | next-themes integration |
| 34 | **Vercel Deployment** | Live URL, custom domain optional |

### Fase 5: Technical Report + Reproducibility Package

| No | Deliverable | Deskripsi |
|----|-------------|-----------|
| 35 | **Paper-Style Technical Report** | `EVALUATION_REPORT.md` — struktur academic |
| 36 | **PDF Version** | Generate PDF dari markdown (pandoc) |
| 37 | **Model Cards** | Setiap model di HF Hub punya model card lengkap |
| 38 | **One-Command Reproduce** | `make train`, `make demo`, dll — full reproducibility |
| 39 | **Blog Post** | Medium/Dev.to article (English + Indonesian) |
| 40 | **Demo Video** | Screen recording 60-90 detik |
| 41 | **GitHub README** | Umbrella README dengan link ke Vercel demo + HF Space + paper + video |

### Fitur Tambahan (Post-MVP, Optional)

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 42 | **Kuzushiji-Kanji Support** | Ekstensi ke 3832 kanji klasik |
| 43 | **Hentaigana Variants** | Handle multiple historical forms per modern character |
| 44 | **Document Type Classifier** | Klasifikasi jenis dokumen |
| 45 | **Batch Processing** | Upload multiple images at once |
| 46 | **Share Results** | Generate shareable image dari result |

## Hal yang Perlu Dipertimbangkan

### Teknis

- **GPU Training Budget**: Kaggle Notebooks 30 jam P100/minggu + Colab 30 jam T4/minggu. Total ~60 jam/minggu gratis
- **Background Execution**: Kaggle mendukung background execution — kritis untuk training 2-4 jam
- **Dataset Size**: Kuzushiji-49 ~50MB, Kaggle Kuzushiji Recognition ~8GB. Keduanya feasible di free tier
- **Class Imbalance**: Kuzushiji-49 tidak balanced (rasio 25:1). WAJIB balanced accuracy + class rebalancing
- **Hentaigana Challenge**: Satu karakter modern bisa 3-5 bentuk kuzushiji berbeda
- **HF Spaces Free Tier**: CPU-only, 16GB RAM, concurrent request ~1-2. Model harus ONNX+quantized
- **Vercel Serverless Limits**: Python runtime, 10 detik timeout, 1GB memory, 250MB package size — cukup untuk proxy pattern
- **Vercel Cold Start**: 2-5 detik pertama setelah idle. Pre-warm bisa dilakukan via Next.js ISR call
- **Character Overlap**: Di kuzushiji, karakter sering overlap — bikin detection lebih sulit
- **Trilingual Gemini API**: Satu foto = 1 API call dengan multi-output prompt (get JP+EN+ID sekaligus)
- **CORS**: Backend Vercel harus allow origin dari frontend Vercel (same domain, tidak masalah) dan dari HF Space (jika ada cross-call)

### Legal & HKI

- **Dataset License**: Kuzushiji-49 dan Kuzushiji Recognition = CC BY-SA 4.0 — gratis untuk riset/produk, attribution ke ROIS-DS
- **Model Weight Ownership**: Trained weights = karya turunan, bisa dilisensikan MIT atau CC BY-SA 4.0
- **HKI Deliverable**: (1) Source code frontend+backend, (2) Trained model weights, (3) Architecture + training recipe, (4) Technical report, (5) UI/UX design
- **Citation Requirements**: WAJIB cite paper Clanuwat et al. 2018 di README, report, model cards

### Privacy & Zero-Storage Architecture

- **NO user data stored**: Foto diproses in-memory di HF Space, tidak pernah disimpan
- **NO tracking/analytics**: Tidak ada Google Analytics, tidak ada cookies tracking
- **NO history**: User tidak bisa lihat upload sebelumnya
- **NO database**: Tidak ada DB di frontend/backend/ML server
- **Transparency**: Privacy notice di UI dalam 3 bahasa
- **GDPR/privacy compliant by default**

### UX Considerations

- **First Impression**: Landing page harus punya hero visual impressive dalam 3 detik pertama
- **Sample Gallery**: 10-15 sample images public domain — user bisa try tanpa upload sendiri
- **Inference Speed**: Target <3 detik per halaman di HF Space CPU tier
- **Error Visualization**: Graceful feedback saat low confidence (warna kuning bukan error)
- **Mobile-First**: Recruiter mungkin coba dari HP
- **Trilingual UX**:
  - Default language: **Japanese**
  - Switcher: 日本語 / English / Bahasa Indonesia
  - Preference stored di localStorage (privacy-safe, bukan cookie)
  - Semua UI strings + error messages + translation output trilingual

---

# B. TECH STACK

## Arsitektur Umum

```
+----------------------------------------------------------+
|                   DATASETS (Free)                          |
|  Kuzushiji-49 (270K classification) + Kaggle Competition  |
|  Kuzushiji Recognition (4K full pages with bounding box)  |
+----------------------------------------------------------+
|                   TRAINING (Free GPU)                      |
|  Kaggle Notebooks (P100 16GB, 30h/week, background exec)  |
|  Google Colab (T4 15GB, 30h/week)                         |
|  PyTorch 2.x + timm + torchvision + Albumentations        |
|  Ultralytics YOLO (detection)                             |
|  Weights & Biases (free tier, experiment tracking)        |
+----------------------------------------------------------+
|                   MODELS                                   |
|  Classification: ResNet-50, ViT-Base (via timm)           |
|  Detection: YOLOv8/YOLO11-medium fine-tuned               |
|  Optimization: ONNX + INT8 quantization                   |
|  Stored on HuggingFace Hub (free, unlimited public)       |
+----------------------------------------------------------+
|                   ML SERVER (HuggingFace Space)            |
|  Gradio SDK 4.x (public demo + auto-generated API)        |
|  Python 3.10+ + ONNX Runtime                              |
|  Trilingual UI (Python dict-based i18n)                   |
|  Gemini API (trilingual translation)                      |
|  Deployed: HuggingFace Spaces (free CPU 16GB)             |
+----------------------------------------------------------+
|                   BACKEND API (Vercel Serverless)          |
|  FastAPI Python runtime                                    |
|  Pydantic (request/response validation)                   |
|  httpx (async HTTP client to HF Space)                    |
|  Rate limiting (in-memory sliding window)                 |
|  Deployed: Vercel Serverless Functions (Python)           |
+----------------------------------------------------------+
|                   FRONTEND (Next.js)                       |
|  Next.js 15 (App Router) + React 19 + TypeScript 5        |
|  Tailwind CSS v4 + shadcn/ui + Framer Motion              |
|  Zustand (client state) + TanStack Query v5 (server)      |
|  next-intl (trilingual) + next-themes (dark/light)        |
|  Lucide React (icons)                                     |
|  Deployed: Vercel Hobby (free 100GB BW, 6K build min/mo)  |
+----------------------------------------------------------+
|                   DOCUMENTATION                            |
|  GitHub (umbrella repo) + HuggingFace Hub (model cards)   |
|  Pandoc (PDF generation) + Medium/Dev.to (blog post)      |
+----------------------------------------------------------+
```

## Detail Tech Stack

### ML Framework & Models

| Teknologi | Versi | Fungsi | Justifikasi |
|-----------|-------|--------|-------------|
| **Python** | 3.10+ | Bahasa ML | Standar industri |
| **PyTorch** | 2.x | DL framework | Industri dominan 2026 |
| **timm** | Latest | Pretrained models | 1000+ vision models |
| **Ultralytics** | YOLOv8/11 | Object detection | 1-command training |
| **torchvision** | 2.x | CV utils | Dataset loaders, transforms |
| **Albumentations** | Latest | Augmentations | Lebih cepat dari torchvision |
| **scikit-learn** | Latest | Metrics | Balanced accuracy, confusion matrix |
| **OpenCV** | Latest | Image preprocessing | Resize, crop, bbox drawing |
| **Pillow** | Latest | Image I/O | Standard Python |
| **ONNX Runtime** | Latest | Inference optimization | 2-3x faster CPU |

### Frontend Stack

| Teknologi | Versi | Fungsi | Justifikasi |
|-----------|-------|--------|-------------|
| **Next.js** | 15 (App Router) | Framework fullstack | Industry standard 2026, RSC, Server Actions |
| **React** | 19 | UI library | Server Components, Suspense, transitions |
| **TypeScript** | 5.x | Type safety | Reduces bugs, IDE autocomplete |
| **Tailwind CSS** | v4 | Utility-first CSS | Rapid dev, konsisten |
| **shadcn/ui** | Latest | Component library | Copy-paste, customizable, accessible |
| **Framer Motion** | 12.x | Animasi | Page transitions, micro-interactions |
| **Zustand** | 5.x | Client state | Lightweight, no Provider wrapper |
| **TanStack Query** | 5.x | Server state | Caching, background refetch |
| **next-intl** | Latest | i18n (trilingual) | Modern, type-safe, RSC-compatible |
| **next-themes** | Latest | Dark/Light mode | SSR-safe theme switching |
| **Lucide React** | Latest | Icons | Tree-shakeable, 1000+ icons |
| **React Dropzone** | Latest | File upload | Drag-drop with validation |

### Backend Stack

| Teknologi | Versi | Fungsi | Justifikasi |
|-----------|-------|--------|-------------|
| **FastAPI** | Latest | Python web framework | Modern, async, auto OpenAPI docs |
| **Pydantic** | 2.x | Data validation | Type-safe request/response |
| **httpx** | Latest | Async HTTP client | Modern, async-native untuk panggil HF Space |
| **python-multipart** | Latest | File upload support | Dibutuhkan FastAPI |

### ML Server (HuggingFace Space) Stack

| Teknologi | Fungsi | Free Tier |
|-----------|--------|-----------|
| **Gradio** | Demo UI + API | Gratis |
| **HuggingFace Spaces** | Hosting | Free CPU 16GB RAM |
| **Gemini API** | Translation | 1000 RPD free |

### Experiment Tracking & MLOps

| Teknologi | Free Tier |
|-----------|-----------|
| **Weights & Biases** | 100GB storage, unlimited runs (personal) |
| **HuggingFace Hub** | Unlimited public models & datasets |
| **Git LFS** | GitHub free: 1GB storage |

### Deployment

| Teknologi | Fungsi | Free Tier |
|-----------|--------|-----------|
| **Vercel Hobby** | Frontend + Serverless Functions | 100GB BW, 6000 build min/mo |
| **HuggingFace Spaces** | ML Server | Free CPU 16GB |

### Development Tools

| Teknologi | Fungsi |
|-----------|--------|
| **Kaggle Notebooks** | Primary training (P100) |
| **Google Colab** | Secondary training (T4) |
| **VS Code + Jupyter** | Local dev |
| **GitHub** | Source control |
| **Ruff + Black** | Python linting/formatting |
| **ESLint + Prettier** | TypeScript/JS linting |
| **pytest** | Python testing |
| **Vitest** | TypeScript testing |

### Documentation

| Teknologi | Fungsi |
|-----------|--------|
| **Pandoc** | Markdown → PDF |
| **Mermaid** | Diagrams |
| **matplotlib + seaborn** | Training plots |
| **OBS Studio** | Demo video |

### Kenapa BUKAN Alternatif Lain?

| Dipertimbangkan | Ditolak Karena |
|----------------|----------------|
| TensorFlow/Keras | Ecosystem lebih kecil di 2026 |
| JAX | Learning curve + community lebih kecil |
| Detectron2 | Over-engineered, YOLOv8 cukup |
| SvelteKit | Ecosystem ~10x lebih kecil dari React |
| Remix / React Router v7 | Vercel integration tidak se-native Next.js |
| Material UI | Lebih enterprise-feel, shadcn lebih modern |
| Supabase / Firebase | Tidak dibutuhkan (zero-storage) |
| Railway / Render | Vercel Serverless cukup untuk proxy pattern |
| Docker containers | Over-engineered, Vercel + HF sudah simple |

---

# C. ARSITEKTUR SISTEM & REPOSITORY

## Repository Structure — Monorepo

```
kobun/                               # Main GitHub repo (monorepo)
├── README.md                        # Umbrella README (bilingual EN+JP)
├── EVALUATION_REPORT.md             # Paper-style ML report
├── EVALUATION_REPORT.pdf            # Generated from markdown
├── CLAUDE.md                        # Progress tracking
├── CHANGELOG.md                     # Version history
├── CITATION.cff                     # Citation metadata
├── LICENSE                          # MIT or CC BY-SA 4.0
├── Makefile                         # Monorepo workflow shortcuts
├── .gitignore
├── .env.example
│
├── apps/
│   ├── web/                         # Next.js 15 frontend + Vercel FastAPI backend
│   │   ├── src/
│   │   │   ├── app/                 # Next.js App Router pages
│   │   │   │   ├── [locale]/        # Trilingual routing (ja/en/id)
│   │   │   │   │   ├── page.tsx     # Landing page
│   │   │   │   │   └── layout.tsx
│   │   │   │   ├── globals.css
│   │   │   │   └── favicon.ico
│   │   │   ├── components/
│   │   │   │   ├── ui/              # shadcn/ui components
│   │   │   │   ├── layout/          # Navbar, Footer, etc
│   │   │   │   ├── upload/          # ImageUploader, Dropzone
│   │   │   │   ├── results/         # ResultViewer, BoundingBoxOverlay
│   │   │   │   └── language/       # LanguageSwitcher
│   │   │   ├── lib/
│   │   │   │   ├── api.ts           # TanStack Query hooks
│   │   │   │   ├── i18n.ts          # next-intl config
│   │   │   │   └── utils.ts         # cn(), helpers
│   │   │   ├── stores/              # Zustand stores
│   │   │   ├── messages/            # next-intl translations
│   │   │   │   ├── ja.json
│   │   │   │   ├── en.json
│   │   │   │   └── id.json
│   │   │   └── types/               # Shared TypeScript types
│   │   ├── api/                     # Vercel Python Serverless Functions
│   │   │   ├── infer.py             # POST /api/infer (proxy ke HF Space)
│   │   │   ├── health.py            # GET /api/health
│   │   │   └── requirements.txt     # Python deps untuk backend
│   │   ├── public/
│   │   │   └── images/              # Static assets (hero, samples thumbnail)
│   │   ├── next.config.ts
│   │   ├── tailwind.config.ts
│   │   ├── tsconfig.json
│   │   ├── package.json
│   │   └── vercel.json              # Vercel config (routes, functions)
│   │
│   └── space/                       # Gradio ML Space (deploy ke HuggingFace)
│       ├── app.py                   # Gradio app entry point
│       ├── i18n.py                  # Trilingual LOCALES dict
│       ├── requirements.txt         # Python deps untuk Space
│       ├── README.md                # HF Space README dengan frontmatter
│       ├── .gitignore
│       └── examples/                # 10-15 public domain sample images
│           ├── sample-01.jpg
│           ├── sample-02.jpg
│           └── ...
│
├── ml/                              # ML training pipeline (training-only)
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── datasets.py          # KuzushijiDataset class
│   │   │   ├── transforms.py        # Augmentations
│   │   │   └── yolo_converter.py    # Bbox format converter
│   │   ├── models/
│   │   │   ├── classifier.py        # ResNet/ViT wrapper
│   │   │   └── detector.py          # YOLO wrapper
│   │   ├── training/
│   │   │   ├── trainer.py           # Main training loop
│   │   │   ├── callbacks.py
│   │   │   └── losses.py
│   │   ├── evaluation/
│   │   │   ├── metrics.py
│   │   │   ├── error_analysis.py
│   │   │   └── visualization.py
│   │   ├── pipeline/
│   │   │   ├── inference.py         # End-to-end pipeline
│   │   │   ├── reading_order.py
│   │   │   └── translation.py       # Gemini trilingual wrapper
│   │   ├── utils/
│   │   │   ├── config.py            # YAML config loader
│   │   │   └── logging.py
│   │   ├── train.py                 # CLI entry point
│   │   ├── evaluate.py
│   │   └── export.py                # ONNX export
│   ├── configs/
│   │   ├── baseline-cnn.yaml
│   │   ├── resnet50-baseline.yaml
│   │   ├── vit-base.yaml
│   │   └── yolov8-detection.yaml
│   ├── notebooks/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_classification_training.ipynb
│   │   ├── 03_detection_training.ipynb
│   │   ├── 04_pipeline_demo.ipynb
│   │   └── 05_error_analysis.ipynb
│   ├── scripts/
│   │   ├── download_data.py         # Kaggle API download
│   │   ├── prepare_detection_data.py
│   │   ├── benchmark_inference.py
│   │   └── generate_report_pdf.py
│   ├── tests/
│   │   ├── test_datasets.py
│   │   ├── test_transforms.py
│   │   └── test_pipeline.py
│   └── requirements.txt             # Python deps untuk training
│
└── .github/
    └── workflows/
        ├── frontend-ci.yml          # Lint/test Next.js on PR
        ├── ml-ci.yml                # Lint/test Python on PR
        └── space-sync.yml           # Auto-sync apps/space/ to HF Space
```

## Deployment Targets

| Component | Source Folder | Deploy Target | URL Pattern |
|-----------|---------------|---------------|-------------|
| **Frontend** | `apps/web/src/` | Vercel Hobby | `kobun-{username}.vercel.app` |
| **Backend API** | `apps/web/api/` | Vercel Serverless (Python) | `{frontend_url}/api/*` |
| **ML Server** | `apps/space/` | HuggingFace Spaces | `huggingface.co/spaces/{username}/kobun` |
| **Models** | `ml/` outputs | HuggingFace Hub | `huggingface.co/{username}/kobun-*` |

## Data Flow

```
1. User uploads image via Next.js frontend
   POST https://kobun-{user}.vercel.app/ (Next.js UI)

2. Frontend calls backend API (same domain, no CORS)
   POST https://kobun-{user}.vercel.app/api/infer
   Body: { image: base64, language: "ja" }

3. Backend (Vercel Serverless Python) validates, rate limits, proxies
   POST https://{user}-kobun.hf.space/api/predict
   Body: { data: [...] }

4. HF Space runs ML pipeline (in-memory)
   - YOLO detection
   - ResNet classification
   - Reading order sort
   - Gemini translation (trilingual)

5. Response chain:
   HF Space → Backend (transform response) → Frontend → User
```

## Data Sync: Monorepo ↔ HuggingFace Space

Karena HuggingFace Space punya git repo terpisah di `huggingface.co/spaces/{user}/kobun`, folder `apps/space/` di GitHub perlu di-sync ke HF Space.

**Pilihan sync method:**

**Option A: Manual push** (untuk MVP)
```bash
# Setelah perubahan di apps/space/
cd apps/space
git remote add space https://huggingface.co/spaces/{user}/kobun
git push space HEAD:main
```

**Option B: GitHub Action auto-sync** (recommended, setelah MVP stabil)
Setup di `.github/workflows/space-sync.yml`:
- Trigger: push ke `main` yang mengubah `apps/space/**`
- Action: sync folder `apps/space/` ke HF Space repo
- Detail lengkap ada di Manual Process Guide

## Environment Separation

| Environment | Frontend URL | Backend URL | HF Space |
|-------------|--------------|-------------|----------|
| **Local Dev** | `localhost:3000` | `localhost:3000/api/*` | (proxy ke prod HF Space) |
| **Production** | `kobun-{user}.vercel.app` | `{frontend}/api/*` | `{user}-kobun.hf.space` |
| **Preview (PR)** | Vercel preview URL | Preview backend | (proxy ke prod HF Space) |

## Secrets Management

| Secret | Disimpan di | Keperluan |
|--------|-------------|-----------|
| `GEMINI_API_KEY` | HF Space Secrets | Translation API |
| `HF_TOKEN` | Local .env, GitHub Actions | Upload models ke Hub |
| `KAGGLE_USERNAME/KEY` | Local .env, Kaggle Notebook | Download datasets |
| `WANDB_API_KEY` | Local .env, Kaggle Notebook | Experiment tracking |
| `HF_SPACE_URL` | Vercel env | Backend memanggil HF Space |

---

# D. DATASET & DATA PIPELINE

## Dataset Utama

### 1. Kuzushiji-49 (Classification)

- **Sumber**: ROIS-DS Center for Open Data in the Humanities (CODH)
- **URL**: https://github.com/rois-codh/kmnist
- **License**: CC BY-SA 4.0
- **Size**: 270,912 images (232,365 train / 38,547 test)
- **Format**: NumPy arrays, 28x28 grayscale
- **Classes**: 49 (48 hiragana + 1 iteration mark)
- **Distribution**: Imbalanced (rasio 25:1)
- **Citation**: Clanuwat et al. 2018

### 2. Kaggle Kuzushiji Recognition (Detection)

- **Sumber**: Kaggle Competition 2019
- **URL**: https://www.kaggle.com/competitions/kuzushiji-recognition
- **License**: CC BY-SA 4.0
- **Size**: ~3,900 full-page images + ~683,464 bounding box annotations
- **Format**: Full-resolution scans (2500-3000px height typical)
- **Classes**: 4,300+ unique characters

### 3. Unicode Mapping Reference

- **Sumber**: CODH `unicode_translation.csv`
- **URL**: https://codh.rois.ac.jp/char-shape/
- **Digunakan untuk**: Kuzushiji → Modern kana mapping

### 4. Sample Images (Public Domain)

Untuk demo gallery (shared antara Next.js frontend dan HF Space):
- **Source 1**: Japan Digital Archives (国立国会図書館デジタルコレクション)
- **Source 2**: Wikipedia Commons (Edo period manuscripts)
- **Count**: 10-15 images
- **Storage**: Full-res di `apps/space/examples/`, thumbnails di `apps/web/public/images/samples/`

## Data Pipeline

### Preprocessing — Classification

```
Raw NumPy arrays (28x28 grayscale)
  |
  +-- Normalize [0, 255] → [0, 1]
  +-- Resize to 32x32/64x64 if needed
  +-- Convert to 3-channel (duplicate) for pretrained models
  |
  v
Train/Val Split (85/15) stratified by class
  |
  +-- Augmentations (train only):
  |   - Random rotation (±10°)
  |   - Random affine
  |   - Cutout / Random erasing
  |   - Mixup / Manifold Mixup
  |
  v
DataLoader (batch_size=128, num_workers=4)
```

### Preprocessing — Detection

```
Full-page scan (variable resolution)
  |
  +-- Resize to 1024x1024 for YOLO
  +-- Parse bbox annotations from CSV
  +-- Convert to YOLO format (normalized)
  |
  v
Class-agnostic simplification:
  - Class 0 = "character"
  - Stage 2 handles per-character classification
  |
  v
Train/Val Split (80/20) by unique image
  |
  +-- Augmentations:
  |   - Mosaic (YOLO standard)
  |   - Horizontal flip DISABLED (text direction matters)
  |   - Color jitter
  |
  v
YOLO DataLoader (Ultralytics managed)
```

### Two-Stage Pipeline

```
Input: Full-page scan
  |
  v
STAGE 1: YOLO Detection (class-agnostic)
  → List of bounding boxes
  |
  v
For each bbox: Crop + pad
  |
  v
STAGE 2: ResNet/ViT Classification (batched)
  → Unicode labels per character
  |
  v
Sort by reading order (top-to-bottom, right-to-left)
  |
  v
STAGE 3: Kuzushiji → Modern kana lookup
  |
  v
STAGE 4: Gemini API (single call, multi-output)
  → Japanese modern + English + Indonesian
  |
  v
Output: Structured JSON with 3 translations + metadata
```

## Data Versioning

- **Raw data**: Downloaded via Kaggle API, not committed
- **Preprocessed data**: HDF5/Parquet cached, not committed
- **Model weights**: HF Hub tags
- **Code**: Git + GitHub

**CRITICAL**: Raw data TIDAK di-commit. Use `ml/scripts/download_data.py`.

---

# E. MODEL ARCHITECTURE & TRAINING PLAN

## Phase 1: Classification Models

### Model 1: Baseline CNN (Warmup)

- **Architecture**: Simple 3-conv + 2-FC (~100K params)
- **Training**: 10 epochs, Adam, lr=1e-3
- **Target**: ~85% balanced accuracy
- **Time**: <10 min P100

### Model 2: ResNet-50 (Main Baseline)

- **Architecture**: ResNet-50 pretrained ImageNet (timm)
- **Modifications**:
  - First conv adapted untuk grayscale
  - Final FC: 1000 → 49 classes
- **Training**: 50 epochs, AdamW, cosine lr (1e-3 → 1e-6)
- **Techniques**: Class-weighted loss, label smoothing 0.1, Mixup α=0.2, Cutout
- **Target**: >96% balanced accuracy
- **Time**: ~1-2h P100

### Model 3: Vision Transformer

- **Architecture**: ViT-Base atau DeiT-Small (timm)
- **Modifications**: Patch size 4x4, input 32x32 atau 64x64
- **Training**: 100 epochs, AdamW, warmup 5 epochs
- **Target**: >97% balanced accuracy
- **Time**: ~2-3h P100

### Ensemble (Optional)

- **Method**: Average logits ResNet-50 + ViT + TTA
- **Target**: >98% balanced accuracy

## Phase 2: Detection Model

### YOLO Fine-tuning

- **Model**: YOLOv8-medium atau YOLO11-medium
- **Pretrained**: COCO weights
- **Task**: Class-agnostic character detection
- **Input size**: 1024x1024
- **Training**: 50-100 epochs
- **Modifications**:
  - Disable horizontal flip
  - Custom anchor boxes (k-means clustering)
  - Focal loss untuk small objects
- **Target**: mAP@0.5 > 0.90
- **Time**: ~4-6h P100

## Training Recipe (Config Example)

```yaml
# ml/configs/resnet50-baseline.yaml
model:
  name: resnet50
  pretrained: true
  num_classes: 49

data:
  dataset: kuzushiji49
  image_size: 28
  batch_size: 128
  num_workers: 4
  augmentations:
    - random_rotation: 10
    - random_erasing: 0.25
    - mixup_alpha: 0.2

training:
  epochs: 50
  optimizer: adamw
  lr: 1e-3
  weight_decay: 1e-4
  scheduler: cosine
  label_smoothing: 0.1
  loss: cross_entropy_balanced

evaluation:
  metric: balanced_accuracy
  test_time_augmentation: true

logging:
  wandb_project: kobun-classification
  save_top_k: 3
  checkpoint_path: ./checkpoints
  hf_hub_repo: {username}/kobun-classifier-resnet50
```

## Hyperparameter Tuning

- **Tool**: W&B Sweeps
- **Method**: Bayesian optimization, 20-30 runs per model

---

# F. EVALUATION METHODOLOGY

## Metrics Utama

### Classification

| Metric | Target |
|--------|--------|
| **Balanced Accuracy** | >96% ResNet-50, >97% ViT |
| **Top-1 Accuracy** | >95% |
| **Top-3 Accuracy** | >99% |
| **Per-class Accuracy** | Semua >85% |
| **Confusion Matrix** | Qualitative analysis |

### Detection

| Metric | Target |
|--------|--------|
| **mAP@0.5** | >0.90 |
| **mAP@0.5:0.95** | >0.65 |
| **Precision** | >0.92 |
| **Recall** | >0.88 |

### End-to-End Pipeline

| Metric | Target |
|--------|--------|
| **Character Accuracy** | >85% |
| **Sequence Accuracy** | >70% |
| **Reading Order Accuracy** | >95% |

### Frontend Performance

| Metric | Target |
|--------|--------|
| **Lighthouse Performance** | >85 |
| **Lighthouse Accessibility** | >95 |
| **Lighthouse SEO** | >95 |
| **Time to First Byte (TTFB)** | <500ms |
| **First Contentful Paint (FCP)** | <1.5s |
| **Largest Contentful Paint (LCP)** | <2.5s |
| **Cumulative Layout Shift (CLS)** | <0.1 |

### Backend API Performance

| Metric | Target |
|--------|--------|
| **Cold start** | <5s |
| **Warm response** | <200ms (excluding HF Space inference) |
| **End-to-end latency** | <5s per image |

## Error Analysis Protocol

1. **Top-10 Worst Classes** — Karakter paling sering salah
2. **Confusion Pairs** — Karakter yang sering tertukar
3. **Failure Case Gallery** — 20 contoh misclassify
4. **Performance by Image Quality**
5. **Hentaigana Analysis**

## Benchmark Comparison

| Method | Balanced Accuracy (K49) | Source |
|--------|------------------------|--------|
| 4-NN Baseline | 86.01% | Paper asli |
| Keras Simple CNN | 89.36% | Paper asli |
| PreActResNet-18 | 96.64% | Paper asli |
| PreActResNet-18 + Manifold Mixup | 97.33% | Paper asli |
| Shake-Shake + Cutout | **98.29%** | Community SOTA |
| **Kobun ResNet-50** | Target: 96-97% | TBD |
| **Kobun ViT** | Target: 97-98% | TBD |

---

# G. FRONTEND DESIGN & UI SPECIFICATION

## Identitas Visual

### Arah Estetika
**"Scholarly Tech Fusion"** — Menggabungkan kesan akademik/literatur klasik (serif typography, aged paper texture subtle, scroll motifs) dengan modernitas tech (gradients, glassmorphism, micro-interactions, dark mode). Hasilnya: platform yang terasa serius untuk preservasi budaya tapi modern untuk tech audience.

### Color Palette

```
BRAND COLORS
  Primary (Deep Indigo)      #1E1B4B   /* Inspired by aged ink */
  Primary Light              #312E81
  Primary Dark               #0F0D2E

  Accent (Vermillion Red)    #DC2626   /* Traditional Japanese red */
  Accent Hover               #EF4444
  Accent Dark                #B91C1C

  Secondary (Washi Beige)    #FEF3C7   /* Traditional paper color */
  Secondary Light            #FFFBEB

  Gold (Ornament)            #D97706   /* Traditional gold accents */
  Gold Light                 #FBBF24

NEUTRALS
  Background Light           #FAFAF9
  Background Dark            #0A0A0A
  Surface Light              #FFFFFF
  Surface Dark               #171717
  Text Primary Light         #171717
  Text Primary Dark          #FAFAFA
  Text Secondary Light       #525252
  Text Secondary Dark        #A3A3A3
  Border Light               #E7E5E4
  Border Dark                #292524

SEMANTIC
  Success                    #10B981
  Error                      #EF4444
  Warning                    #F59E0B
  Info                       #3B82F6

CONFIDENCE COLORS (untuk bounding box)
  High (>90%)                #10B981
  Medium (70-90%)            #F59E0B
  Low (<70%)                 #EF4444
```

### Typography

| Usage | Font | Weight | Size | Rationale |
|-------|------|--------|------|-----------|
| **Display / Hero** | **Fraunces** | 700, 900 | 48-96px | Modern serif dengan historic feel |
| **Section Heading** | **Inter** | 600, 700 | 24-36px | Clean geometric sans-serif |
| **Body** | **Inter** | 400, 500 | 14-16px | Excellent readability |
| **Japanese Text** | **Noto Serif JP** | 400, 700 | 18-48px | Serif JP untuk kesan klasik |
| **Modern JP Output** | **Noto Sans JP** | 400, 500 | 16-24px | Modern sans untuk transcription |
| **Monospace** | **JetBrains Mono** | 400 | 13-14px | Unicode codes, technical |

Rationale pemilihan font berbeda dari Kioku: Kobun = "scholarly/historical", Kioku = "friendly/educational". Distinct tapi sama-sama profesional.

### Spacing & Radius

| Token | Value | Usage |
|-------|-------|-------|
| `space-1` | 4px | Micro gap |
| `space-2` | 8px | Tight |
| `space-3` | 12px | Compact |
| `space-4` | 16px | Default |
| `space-6` | 24px | Comfortable |
| `space-8` | 32px | Section |
| `space-12` | 48px | Large section |
| `space-16` | 64px | Hero |
| `space-24` | 96px | Major section |

| Radius | Value |
|--------|-------|
| `sm` | 6px |
| `md` | 12px |
| `lg` | 16px |
| `xl` | 24px |
| `2xl` | 32px |
| `full` | 9999px |

## Page Layout

### Landing Page (`/`)

```
+--------------------------------------------------------------------+
| [Kobun Logo]  Try It  About  Models  Paper   [日本語 ▼]  [🌙/☀️] |
+--------------------------------------------------------------------+
|                                                                    |
|                    HERO SECTION                                    |
|                                                                    |
|   [Historical document image with animated AI overlay]             |
|                                                                    |
|           Read Classical Japanese with AI                          |
|           古文書をAIで解読する                                     |
|                                                                    |
|      "Less than 0.01% of Japanese people can read kuzushiji.       |
|       Kobun changes that using deep learning."                     |
|                                                                    |
|       [Try It Free →]         [View on GitHub →]                   |
|                                                                    |
+--------------------------------------------------------------------+
|                    TRY IT SECTION                                  |
|                                                                    |
|   [Drag & drop image here, or click to browse]                     |
|                                                                    |
|   Or try a sample:                                                 |
|   [sample1] [sample2] [sample3] [sample4] [sample5]                |
|   [sample6] [sample7] [sample8] [sample9] [sample10]               |
|                                                                    |
|   Your images are never stored or shared.                          |
+--------------------------------------------------------------------+
|                    RESULTS SECTION (after upload)                  |
|                                                                    |
|   [Input image] [→] [Annotated output with bounding boxes]         |
|                                                                    |
|   Transcription (Modern Japanese):                                 |
|   わたしはにほんじんです                                           |
|                                                                    |
|   Translation ([selected language]):                               |
|   Saya adalah orang Jepang.                                        |
|                                                                    |
|   Characters: 12  •  Avg confidence: 94%  •  Time: 2.3s            |
|                                                                    |
|   [Try another image]  [Copy result]  [Share]                      |
+--------------------------------------------------------------------+
|                    HOW IT WORKS                                    |
|                                                                    |
|   [Step 1: Detect]  [Step 2: Classify]  [Step 3: Translate]        |
+--------------------------------------------------------------------+
|                    TECHNICAL DETAILS                               |
|                                                                    |
|   [ResNet-50: 97%]  [ViT: 97.8%]  [YOLO: mAP 0.92]                 |
|                                                                    |
|   [Download Models]  [Read Paper]  [Clone Repo]                    |
+--------------------------------------------------------------------+
|  FOOTER: About · GitHub · HF Space · Paper · Citation · License    |
+--------------------------------------------------------------------+
```

### Responsive Breakpoints

```
MOBILE (<768px):
  - Hero single column
  - Upload zone full-width
  - Results stacked vertically
  - Navigation jadi hamburger menu

TABLET (768-1023px):
  - Hero still single column
  - Upload + Results side-by-side if fits
  - Full nav visible

DESKTOP (>=1024px):
  - Hero multi-column with decorative elements
  - Upload + Results side-by-side (60/40 split)
  - Full nav + right-aligned actions
```

## Komponen yang Perlu Dibuat

### Layout Components
- `Navbar` — Top navigation (logo, menu, language switcher, theme toggle)
- `Footer` — Links, citation, license
- `Container` — Max-width responsive wrapper
- `PageWrapper` — Main layout dengan header + footer

### Hero & Landing
- `HeroSection` — Hero dengan animated overlay
- `SampleGallery` — Grid of sample thumbnails
- `HowItWorksSection` — 3-step visualization
- `TechnicalDetailsSection` — Model metrics showcase
- `CTASection` — Final call-to-action

### Upload & Processing
- `ImageDropzone` — Drag-drop zone dengan file validation
- `ImagePreview` — Shows uploaded image
- `UploadProgress` — Progress bar + stage indicator ("Detecting... Classifying... Translating...")
- `ErrorMessage` — Graceful error display

### Results
- `ResultsContainer` — Wrapper untuk result section
- `AnnotatedImage` — Image dengan bounding box overlay
- `BoundingBoxOverlay` — SVG overlay atas image, colored by confidence
- `TranscriptionCard` — Modern Japanese output
- `TranslationCard` — Selected language output
- `MetadataBar` — Character count, confidence, time
- `ActionButtons` — Copy, Share, Try Another

### Shared
- `LanguageSwitcher` — Dropdown/radio untuk JP/EN/ID
- `ThemeToggle` — Dark/light mode switch
- `LoadingSpinner`
- `EmptyState`
- `Toaster` (untuk feedback)

## Animations (Framer Motion)

- **Page Enter**: Fade in + slight slide up
- **Hero Image**: Subtle parallax + AI overlay line animation
- **Sample Gallery**: Stagger reveal on scroll
- **Upload Zone**: Pulse on hover, expand on drag-over
- **Processing**: Smooth progress bar with step indicators
- **Results Reveal**: Scale + fade in, sequential reveal of elements
- **Bounding Boxes**: Sequentially draw on annotated image (SVG stroke animation)
- **Hover States**: Subtle scale + shadow changes

## Dark Mode

- Full dark mode support via `next-themes`
- SSR-safe, no FOUC
- Respects system preference by default
- User preference saved in localStorage

## Trilingual Implementation

**Library**: `next-intl` (modern, type-safe, RSC-compatible)

**Structure**:
```
apps/web/src/messages/
├── ja.json        # Default
├── en.json
└── id.json
```

**Route Structure**:
```
/ja/                 (default, / redirects here)
/en/
/id/
```

**Example**:
```json
// apps/web/src/messages/ja.json
{
  "hero": {
    "title": "古文書をAIで解読する",
    "subtitle": "AIを使って、くずし字で書かれた古典文学を読解します",
    "cta": "今すぐ試す"
  }
}

// apps/web/src/messages/en.json
{
  "hero": {
    "title": "Read Classical Japanese with AI",
    "subtitle": "Decode kuzushiji classical literature using deep learning",
    "cta": "Try It Now"
  }
}

// apps/web/src/messages/id.json
{
  "hero": {
    "title": "Baca Jepang Klasik dengan AI",
    "subtitle": "Pecahkan kuzushiji literatur klasik dengan deep learning",
    "cta": "Coba Sekarang"
  }
}
```

---

# H. BACKEND API & INTEGRATION

## API Architecture

Backend adalah **lightweight proxy** antara frontend dan HF Space. Tidak ada business logic berat — hanya validation, rate limiting, dan forward request.

## Endpoints

### `POST /api/infer`

**Purpose**: Submit image untuk inference

**Request**:
```json
{
  "image": "base64-encoded-image-string",
  "language": "ja"
}
```

**Validation** (Pydantic):
- `image`: base64 string, decoded size <5MB
- `language`: enum (ja/en/id)
- `content-type`: image/png, image/jpeg, image/webp only

**Rate Limiting**:
- 10 requests/minute per IP
- 429 response kalau exceeded
- In-memory sliding window

**Response (200)**:
```json
{
  "success": true,
  "data": {
    "annotated_image": "base64-encoded-output",
    "transcription": "わたしはにほんじんです",
    "translations": {
      "ja": "私は日本人です。",
      "en": "I am Japanese.",
      "id": "Saya orang Jepang."
    },
    "characters": [
      {
        "bbox": [10, 20, 50, 60],
        "label": "わ",
        "unicode": "U+308F",
        "confidence": 0.98
      }
    ],
    "metadata": {
      "character_count": 12,
      "avg_confidence": 0.94,
      "processing_time_ms": 2340
    }
  }
}
```

**Response (Error)**:
```json
{
  "success": false,
  "error": {
    "code": "IMAGE_TOO_LARGE",
    "message": "Image exceeds 5MB limit"
  }
}
```

**Error Codes**:
- `INVALID_IMAGE` — File bukan image atau corrupt
- `IMAGE_TOO_LARGE` — Melebihi 5MB
- `RATE_LIMIT_EXCEEDED` — 429
- `HF_SPACE_UNAVAILABLE` — HF Space down atau timeout
- `INFERENCE_ERROR` — Error saat ML inference
- `INTERNAL_ERROR` — Fallback 500

### `GET /api/health`

**Purpose**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "hf_space_reachable": true,
  "version": "2.0.0"
}
```

## Backend Implementation Pattern

```python
# apps/web/api/infer.py (Vercel Serverless Function)
from pydantic import BaseModel, Field, field_validator
import httpx
import base64
import os

HF_SPACE_URL = os.getenv("HF_SPACE_URL")
RATE_LIMIT_WINDOW = {}

class InferRequest(BaseModel):
    """Validated request body for inference endpoint."""
    
    image: str = Field(..., description="Base64-encoded image")
    language: str = Field("ja", pattern="^(ja|en|id)$")

    @field_validator("image")
    @classmethod
    def validate_image_size(cls, v: str) -> str:
        decoded_size = len(base64.b64decode(v))
        if decoded_size > 5 * 1024 * 1024:
            raise ValueError("Image exceeds 5MB")
        return v


async def handler(request):
    """Handle inference request by proxying to HF Space."""
    client_ip = request.headers.get("x-forwarded-for", "unknown")
    
    if is_rate_limited(client_ip):
        return error_response("RATE_LIMIT_EXCEEDED", 429)
    
    try:
        body = InferRequest.model_validate_json(request.body)
    except Exception as e:
        return error_response("INVALID_IMAGE", 400, str(e))
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{HF_SPACE_URL}/run/predict",
                json={"data": [body.image, body.language]}
            )
            response.raise_for_status()
            return success_response(response.json())
        except httpx.TimeoutException:
            return error_response("HF_SPACE_UNAVAILABLE", 503)
        except Exception as e:
            return error_response("INFERENCE_ERROR", 500, str(e))
```

## CORS Configuration

Karena frontend dan backend di domain yang sama (Vercel), CORS tidak bermasalah untuk request dari frontend. Tapi tetap perlu:

- Strict origin check untuk `/api/infer` endpoint
- Optional: shared secret untuk verify request origin

## Environment Variables (Backend)

```bash
HF_SPACE_URL=https://{username}-kobun.hf.space
BACKEND_VERSION=2.0.0
```

---

# I. ML SERVER (HUGGINGFACE SPACE)

## Role Ganda

1. **Public ML Demo** — User bisa langsung akses `huggingface.co/spaces/{user}/kobun` untuk quick-try tanpa buka website custom
2. **API Endpoint** — Dipanggil oleh backend Vercel FastAPI untuk handle inference dari frontend custom

## Space Configuration

```yaml
# apps/space/README.md frontmatter
title: Kobun - Kuzushiji AI Reader
emoji: 📜
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.x
app_file: app.py
pinned: true
license: cc-by-sa-4.0
models:
  - {username}/kobun-classifier-resnet50
  - {username}/kobun-classifier-vit
  - {username}/kobun-detector-yolov8
datasets:
  - rois-codh/kmnist
```

## Hardware Requirements

- **Tier**: Free CPU (16GB RAM, 50GB storage)
- **Optimization**: ONNX + INT8 quantization, inference <3s per image

## Zero-Storage Architecture

Image processed in-memory, tidak pernah disimpan. Yang tidak dilakukan:
- `image.save()`
- Database connection
- Logging image bytes
- Analytics tracking

## Trilingual UI (Gradio)

```python
# apps/space/i18n.py
LOCALES = {
    "ja": {
        "title": "Kobun - 古文書AIリーダー",
        "subtitle": "くずし字をAIで解読",
        "upload_label": "画像をアップロード",
        "privacy_notice": "あなたの画像は保存されません",
        "transcription_label": "翻刻 (現代日本語)",
        "translation_label": "翻訳",
        "stats_label": "統計",
        "processing": "処理中...",
    },
    "en": {
        "title": "Kobun - Classical Japanese AI Reader",
        "subtitle": "Decode kuzushiji with AI",
        "upload_label": "Upload image",
        "privacy_notice": "Your images are never stored",
        "transcription_label": "Transcription (Modern Japanese)",
        "translation_label": "Translation",
        "stats_label": "Stats",
        "processing": "Processing...",
    },
    "id": {
        "title": "Kobun - Pembaca Jepang Klasik dengan AI",
        "subtitle": "Baca kuzushiji dengan AI",
        "upload_label": "Unggah gambar",
        "privacy_notice": "Gambar Anda tidak disimpan",
        "transcription_label": "Transkripsi (Jepang Modern)",
        "translation_label": "Terjemahan",
        "stats_label": "Statistik",
        "processing": "Memproses...",
    }
}

DEFAULT_LANGUAGE = "ja"
```

## Gradio App Structure

```python
# apps/space/app.py
"""Kobun Gradio demo - zero-storage trilingual ML server."""

import gradio as gr
from PIL import Image

from i18n import LOCALES, DEFAULT_LANGUAGE
from pipeline_wrapper import KobunPipeline


pipeline = KobunPipeline.from_pretrained()


def process(image: Image.Image, language: str) -> tuple:
    """Process uploaded image in-memory.
    
    Args:
        image: User-uploaded PIL Image
        language: Selected UI language (ja/en/id)
    
    Returns:
        Tuple of (annotated_image, transcription, translation, stats)
    """
    result = pipeline.infer(image)
    translation = result.translations[language]
    
    locale = LOCALES[language]
    stats = (
        f"{locale['stats_chars']}: {len(result.characters)} | "
        f"{locale['stats_confidence']}: {result.avg_confidence:.1%} | "
        f"{locale['stats_time']}: {result.time_ms}ms"
    )
    
    return result.annotated_image, result.transcription, translation, stats


def update_ui(language: str) -> dict:
    """Re-render UI labels when language changes."""
    locale = LOCALES[language]
    return {
        title_md: locale["title"],
        upload_img: gr.update(label=locale["upload_label"]),
        transcription_box: gr.update(label=locale["transcription_label"]),
        translation_box: gr.update(label=locale["translation_label"]),
        stats_box: gr.update(label=locale["stats_label"]),
    }


with gr.Blocks(title="Kobun") as demo:
    language = gr.Radio(
        choices=["ja", "en", "id"],
        value=DEFAULT_LANGUAGE,
        label="Language / 言語 / Bahasa"
    )
    
    title_md = gr.Markdown(LOCALES[DEFAULT_LANGUAGE]["title"])
    
    with gr.Row():
        upload_img = gr.Image(
            type="pil",
            label=LOCALES[DEFAULT_LANGUAGE]["upload_label"]
        )
        output_img = gr.Image(label="Detected")
    
    transcription_box = gr.Textbox(
        label=LOCALES[DEFAULT_LANGUAGE]["transcription_label"]
    )
    translation_box = gr.Textbox(
        label=LOCALES[DEFAULT_LANGUAGE]["translation_label"]
    )
    stats_box = gr.Textbox(label=LOCALES[DEFAULT_LANGUAGE]["stats_label"])
    
    language.change(
        fn=update_ui,
        inputs=language,
        outputs=[title_md, upload_img, transcription_box,
                 translation_box, stats_box]
    )
    
    upload_img.change(
        fn=process,
        inputs=[upload_img, language],
        outputs=[output_img, transcription_box, translation_box, stats_box]
    )


if __name__ == "__main__":
    demo.launch()
```

## Auto-Generated API Endpoint

Gradio otomatis expose REST API endpoint:

```
POST https://{username}-kobun.hf.space/run/predict
Content-Type: application/json

Body:
{
  "data": [
    "base64-image-string",
    "ja"
  ]
}

Response:
{
  "data": [
    "base64-annotated-image",
    "transcription",
    "translation",
    "stats"
  ]
}
```

Backend FastAPI memanggil endpoint ini, kemudian transform response ke format yang lebih clean untuk frontend.

## Model Optimization

### ONNX Export

```python
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=17,
    dynamic_axes={'input': {0: 'batch'}}
)

from onnxruntime.quantization import quantize_dynamic
quantize_dynamic("model.onnx", "model_quantized.onnx")
```

Expected speedup: 2-3x pada CPU.

### Caching Strategy

- **Model loading**: Lazy load at startup, keep in memory
- **Preprocessing**: Reuse transforms
- **Translation caching**: In-memory LRU cache (5 min TTL) untuk sample images

---

# J. TECHNICAL REPORT & REPRODUCIBILITY

## Technical Report Structure

File: `EVALUATION_REPORT.md` (+ PDF via pandoc)

Struktur academic:

1. **Abstract** (1 paragraf)
2. **Introduction** — Cultural context, motivation, contributions
3. **Related Work** — Clanuwat 2018, Kaggle solutions, modern OCR
4. **Dataset Description** — Stats, imbalance, samples
5. **Methodology** — Arsitektur, training, augmentations
6. **Results** — Metrics, curves, confusion matrix
7. **Error Analysis** — Worst classes, confusion pairs, failure gallery
8. **System Architecture** (unik untuk v2.0) — Full-stack deployment pipeline
9. **Discussion** — Architecture comparisons, bottlenecks
10. **Limitations** — 49 classes, hentaigana, translation quality
11. **Future Work** — Kuzushiji-Kanji, TrOCR, multi-modal
12. **Reproducibility** — One-command instructions
13. **References**
14. **Acknowledgments**

## Model Cards (HuggingFace Hub)

Setiap model di HF Hub punya model card:

```markdown
---
language: ja
license: cc-by-sa-4.0
tags:
- vision
- image-classification
- japanese
- kuzushiji
- cultural-heritage
datasets:
- rois-codh/kmnist
metrics:
- balanced_accuracy
base_model: timm/resnet50.a1_in1k
---

# Kobun Classifier (ResNet-50)

## Model Description
## Intended Use
## Training Data
## Training Procedure
## Evaluation Results
## Limitations and Bias
## Environmental Impact
## Citation
```

## One-Command Reproducibility

```bash
# Clone + setup
git clone https://github.com/{username}/kobun
cd kobun

# ML Training (via Makefile)
make setup           # Setup Python env
make data            # Download datasets
make train-all       # Train all models
make evaluate        # Run evaluations
make export          # Convert to ONNX

# Local development
make dev-frontend    # Run Next.js dev server
make dev-backend     # Run FastAPI locally
make dev-space       # Run Gradio Space locally

# Deployment
make deploy-space    # Sync apps/space/ to HF Space
# Frontend + Backend auto-deploy via Vercel on git push

# Documentation
make report          # Generate EVALUATION_REPORT.pdf
```

## Blog Post Strategy

- **Platform**: Medium (primary) + Dev.to (cross-post)
- **Bilingual**: English (utama) + Indonesian (secondary)
- **Length**: 1500-2500 words
- **Structure**: Hook → Context → Technical Journey → Architecture → Results → Lessons → CTA

## Demo Video

- **Duration**: 60-90 detik
- **Platform**: YouTube (Unlisted) + embed di README
- **Tool**: OBS Studio
- **Structure**:
  - 0-10s: Problem statement
  - 10-40s: Live demo — custom UI (Vercel) + HF Space comparison
  - 40-60s: Technical architecture overview
  - 60-90s: CTA (demo URLs, GitHub, paper)

---

# K. PROJECT PHASES & DELIVERABLES

## MANUAL SETUP — WAJIB SEBELUM PHASE 1

Detail lengkap di Bagian L → Manual Process Guide. Ringkasan:

- Kaggle account + API key
- Weights & Biases account + API key
- HuggingFace account + access token (write)
- GitHub repo "kobun" (monorepo)
- Gemini API key
- **Vercel account** (untuk frontend + backend deployment)
- Local: Python 3.10+, Node.js 20+, pnpm, git, VS Code

## Phase 1: Classification (Week 1-2)

**Goals**: Trained classification models + reproducible training pipeline

### Manual Steps
1. Setup Kaggle API di lokal (Bagian L)
2. Create W&B project "kobun-classification"
3. Create HF Hub repos: baseline, resnet50, vit

### Tasks
- [ ] Setup `ml/` folder structure
- [ ] `ml/src/data/datasets.py` — KuzushijiDataset
- [ ] `ml/src/models/classifier.py` — ResNet/ViT wrappers
- [ ] `ml/src/training/trainer.py` — Training loop + W&B
- [ ] `ml/src/training/losses.py` — Class-weighted, mixup, label smoothing
- [ ] `ml/src/evaluation/metrics.py` — Balanced accuracy, confusion matrix
- [ ] `ml/src/train.py` — CLI entry
- [ ] `ml/scripts/download_data.py` — Kaggle API download
- [ ] `ml/configs/*.yaml` — 3 training configs
- [ ] `ml/notebooks/01_eda.ipynb`
- [ ] Train baseline CNN
- [ ] Train ResNet-50 (target >96%)
- [ ] Train ViT-Base (target >97%)
- [ ] W&B sweep (20-30 runs)
- [ ] Upload models ke HF Hub dengan model cards (draft)
- [ ] `EVALUATION_REPORT.md` partial (Phase 1 section)

### Deliverables
- `ml/` folder lengkap
- 3 trained models di HF Hub
- `EVALUATION_REPORT.md` partial

### Success Criteria
- ResNet-50 >96% balanced accuracy
- ViT >97% balanced accuracy
- Training reproducible dari config

## Phase 2: Detection + ML Pipeline (Week 3-5)

**Goals**: Full-page detection + end-to-end ML pipeline

### Manual Steps
1. Download Kaggle Kuzushiji Recognition dataset
2. Create W&B project "kobun-detection"
3. Setup Gemini API key
4. Create HF Hub repo `kobun-detector-yolov8`

### Tasks
- [ ] `ml/src/data/yolo_converter.py`
- [ ] `ml/configs/yolov8-detection.yaml`
- [ ] YOLO training (target mAP@0.5 >0.90)
- [ ] `ml/src/pipeline/inference.py` — Two-stage pipeline
- [ ] `ml/src/pipeline/reading_order.py`
- [ ] `ml/src/pipeline/translation.py` — Gemini trilingual wrapper
- [ ] Test pipeline end-to-end
- [ ] Upload YOLO ke HF Hub
- [ ] `ml/notebooks/02_detection_training.ipynb`
- [ ] `ml/notebooks/03_pipeline_demo.ipynb`
- [ ] `EVALUATION_REPORT.md` Phase 2 section

### Deliverables
- Trained YOLO + model card
- End-to-end pipeline working
- Trilingual translation layer

### Success Criteria
- mAP@0.5 >0.90
- Character accuracy >85%
- Trilingual output working

## Phase 3: ML Server (HuggingFace Space) (Week 6)

**Goals**: Live zero-storage Gradio demo + API endpoint

### Manual Steps
1. Create HuggingFace Space "kobun"
2. Add GEMINI_API_KEY secret
3. Setup HF Space git remote
4. Curate 10-15 public domain sample images

### Tasks
- [ ] `ml/src/export.py` — ONNX + quantization
- [ ] `apps/space/app.py` — Gradio app
- [ ] `apps/space/i18n.py` — Trilingual LOCALES
- [ ] `apps/space/examples/` — Sample images
- [ ] `apps/space/requirements.txt`
- [ ] `apps/space/README.md` — Bilingual EN+JP frontmatter
- [ ] Initial push ke HF Space
- [ ] Test inference latency
- [ ] Test trilingual UI
- [ ] Test auto-generated API endpoint

### Deliverables
- Public URL: `https://{username}-kobun.hf.space`
- Working trilingual demo
- Auto-generated API callable

### Success Criteria
- Inference <3s per image
- Trilingual switcher working
- API endpoint responds correctly

## Phase 4: Custom Frontend + Backend (Week 7-10)

**Goals**: Premium Next.js UI + FastAPI backend

### Manual Steps
1. Create Vercel account + connect GitHub
2. Create new Vercel project linked to monorepo
3. Configure Vercel root directory ke `apps/web/`
4. Set environment variables di Vercel

### Tasks — Week 7 (Setup + Core)
- [ ] Setup `apps/web/` Next.js 15 + TypeScript
- [ ] Configure Tailwind CSS v4
- [ ] Install dan setup shadcn/ui
- [ ] Install Framer Motion, Zustand, TanStack Query
- [ ] Setup next-intl (trilingual routing)
- [ ] Setup next-themes (dark mode)
- [ ] Create `apps/web/src/messages/{ja,en,id}.json`
- [ ] Layout components: Navbar, Footer, Container
- [ ] LanguageSwitcher + ThemeToggle

### Tasks — Week 8 (Landing Page)
- [ ] HeroSection component
- [ ] SampleGallery component
- [ ] HowItWorksSection component
- [ ] TechnicalDetailsSection component
- [ ] Footer content
- [ ] Responsive design (mobile/tablet/desktop)
- [ ] Framer Motion page transitions

### Tasks — Week 9 (Upload & Results + Backend)
- [ ] `apps/web/api/infer.py` — FastAPI proxy ke HF Space
- [ ] `apps/web/api/health.py` — Health check
- [ ] Pydantic schemas untuk request/response
- [ ] Rate limiting (in-memory sliding window)
- [ ] ImageDropzone component (react-dropzone)
- [ ] UploadProgress component
- [ ] ResultsContainer + AnnotatedImage + BoundingBoxOverlay
- [ ] TranscriptionCard + TranslationCard
- [ ] MetadataBar + ActionButtons
- [ ] TanStack Query setup untuk API calls
- [ ] Error handling + toast notifications

### Tasks — Week 10 (Polish + Deploy)
- [ ] Test full flow end-to-end (local)
- [ ] Deploy ke Vercel (first deployment)
- [ ] Test production (frontend + backend + HF Space)
- [ ] Fix CORS issues if any
- [ ] Lighthouse optimization (performance, accessibility, SEO)
- [ ] Open Graph meta tags + favicon
- [ ] Setup custom domain (optional)
- [ ] `EVALUATION_REPORT.md` Phase 3+4 section (System Architecture)

### Deliverables
- Live URL: `https://kobun-{username}.vercel.app`
- Working full flow (upload → backend → HF Space → results)
- Lighthouse scores all >85
- Trilingual UI working
- Dark/light mode working

### Success Criteria
- End-to-end latency <5s
- Lighthouse Performance >85
- Lighthouse Accessibility >95
- Mobile responsive
- Zero critical errors production

## Phase 5: Technical Report + Reproducibility (Week 11-12)

**Goals**: Paper-style report + promotion materials

### Manual Steps
1. Install pandoc di lokal
2. Create/check Medium + Dev.to accounts
3. Setup OBS for video recording
4. Create YouTube account (jika belum)

### Tasks
- [ ] Finalize `EVALUATION_REPORT.md` (all sections complete)
- [ ] Generate PDF via pandoc
- [ ] Complete model cards di HF Hub (4 models)
- [ ] `Makefile` lengkap dengan semua targets
- [ ] Test reproduce-from-scratch (clean env)
- [ ] Record demo video (60-90s)
- [ ] Upload video ke YouTube
- [ ] Write blog post (English)
- [ ] Write blog post (Indonesian, optional)
- [ ] Finalize umbrella `README.md` (bilingual EN+JP)
- [ ] `CITATION.cff`
- [ ] `CHANGELOG.md`
- [ ] Publish HF Space as public
- [ ] Flip HF Hub models to public

### Deliverables
- `EVALUATION_REPORT.md` + `.pdf`
- 4 complete model cards
- Demo video on YouTube
- Blog post published
- Final umbrella README
- Full reproducibility verified

### Success Criteria
- Stranger bisa reproduce <2 jam
- Technical report paper-quality
- Blog post >100 views week 1

## Target Estimasi Keseluruhan

**9-12 minggu** part-time (10-15 jam/minggu)

Timeline breakdown:
- Phase 1: 1-2 minggu
- Phase 2: 2-3 minggu
- Phase 3: 1 minggu
- Phase 4: 3-4 minggu (major frontend work)
- Phase 5: 1-2 minggu

---

# L. ATURAN CODING

## Prinsip Umum

1. **Bahasa Kode**: Semua nama variabel, fungsi, class, file, comment, commit message dalam **Bahasa Inggris**
2. **Bahasa User-Facing**:
   - Frontend UI (Next.js): **Trilingual (JP default, EN, ID)** via next-intl
   - ML Server UI (Gradio): **Trilingual** via Python dict
   - Translation output: **Trilingual**
   - README GitHub umbrella: **Bilingual (English + Japanese)**
   - HF Space README: **Bilingual (English + Japanese)**
   - Blog post: **English (utama)**, **Indonesian (secondary)**
   - Technical report: **English**
3. **ZERO emoticon/emoji di source code**: Dilarang keras di Python, TypeScript, React components, markdown technical (spec, CLAUDE.md, report). Emoji hanya boleh di public README untuk visual appeal
4. **Comment minimalis**:
   - Comment HANYA saat benar-benar perlu (algoritma kompleks, rationale keputusan non-obvious)
   - JANGAN tambahkan comment yang cuma menjelaskan ulang kode
   - Function signature + docstring > banyak comment inline
5. **Type hints / TypeScript strict**:
   - Python: Type hints wajib di production code (src/). Lebih longgar di notebooks
   - TypeScript: `strict: true` di tsconfig.json. No `any` tanpa justifikasi
6. **Docstrings / JSDoc**: Wajib untuk non-trivial functions. Short & informative

### Contoh Comment yang BURUK (hindari)

```python
# Set learning rate
lr = 1e-3

# Loop through epochs
for epoch in range(num_epochs):
    # Train one epoch
    train_one_epoch(model, dataloader)
```

```tsx
// Get user from store
const user = useUserStore();

// Check if loaded
if (isLoading) {
  return <Loading />;
}
```

Semua redundant. Kode sudah self-explanatory.

### Contoh Comment yang BAIK

```python
def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced classification.
    
    Uses inverse frequency weighting with sqrt normalization
    to avoid extreme weights on rare classes (Clanuwat 2018 finding).
    
    Args:
        labels: Training labels, shape (N,)
    
    Returns:
        Weights tensor, shape (num_classes,)
    """
    counts = np.bincount(labels)
    # Sqrt normalization follows Clanuwat 2018 Section 4.2
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)
```

Comment menjelaskan WHY, bukan WHAT.

## Konvensi Penamaan

| Jenis | Konvensi | Contoh |
|-------|---------|--------|
| Python file/folder | snake_case | `train_classifier.py` |
| TypeScript file | kebab-case | `image-dropzone.tsx`, `use-infer.ts` |
| React component | PascalCase | `ImageDropzone`, `ResultsContainer` |
| Python class | PascalCase | `KuzushijiDataset` |
| Python function/var | snake_case | `train_one_epoch` |
| TS function/var | camelCase | `handleUpload`, `isLoading` |
| Constant | UPPER_SNAKE_CASE | `NUM_CLASSES = 49` |
| Jupyter notebook | `NN_snake_case.ipynb` | `01_eda.ipynb` |
| Config file | kebab-case | `resnet50-baseline.yaml` |
| Model checkpoint | `{model}_{dataset}_{metric:.4f}_{epoch}.pt` | `resnet50_k49_0.9712_e42.pt` |
| HF Hub model | `kobun-{type}-{arch}` | `kobun-classifier-resnet50` |
| Env variable | UPPER_SNAKE_CASE | `GEMINI_API_KEY` |
| i18n key | camelCase dot-separated | `hero.title`, `upload.dropzone.label` |

## Frontend Coding Standards

### Component Pattern

```tsx
// apps/web/src/components/upload/image-dropzone.tsx
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import { useTranslations } from "next-intl";

import { cn } from "@/lib/utils";

interface ImageDropzoneProps {
  onFileAccepted: (file: File) => void;
  maxSize?: number;
  className?: string;
}

export function ImageDropzone({
  onFileAccepted,
  maxSize = 5 * 1024 * 1024,
  className,
}: ImageDropzoneProps) {
  const t = useTranslations("upload");
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    maxSize,
    onDrop: (files) => onFileAccepted(files[0]),
  });

  return (
    <motion.div
      {...getRootProps()}
      className={cn("dropzone-base", isDragActive && "dropzone-active", className)}
      whileHover={{ scale: 1.02 }}
    >
      <input {...getInputProps()} />
      <p>{t("dropzone.label")}</p>
    </motion.div>
  );
}
```

**Patterns**:
- Function declaration untuk exported components
- Props interface above component
- Imports: external → internal libs → components → types → styles
- Use `cn()` helper untuk className composition
- useTranslations() untuk i18n

### Server Component vs Client Component

- **Default**: Server Components (no "use client" directive)
- **Client Components**: Only saat butuh interactivity (useState, useEffect, event handlers, Framer Motion)
- Letakkan "use client" di top file

### API Responses

- Semua API responses: `{ success: boolean, data?: T, error?: {code, message} }`
- Zod validation di frontend (atau Pydantic di backend)

## Backend (FastAPI) Coding Standards

### Endpoint Pattern

```python
# apps/web/api/infer.py
"""Inference endpoint - proxies requests to HuggingFace Space."""

from pydantic import BaseModel, Field
import httpx


class InferRequest(BaseModel):
    """Validated request body for inference endpoint."""
    
    image: str = Field(..., description="Base64-encoded image")
    language: str = Field("ja", pattern="^(ja|en|id)$")


async def handler(request):
    """Handle inference request by proxying to HF Space.
    
    Flow:
    1. Validate request body with Pydantic
    2. Check rate limit for client IP
    3. Forward to HF Space inference endpoint
    4. Transform response to frontend-friendly format
    """
    pass
```

## ML Pipeline Coding Standards

### Config-Driven Training

Semua training dijalankan via config YAML. Tidak ada hardcoded hyperparameters di code.

### Reproducibility

- Set random seeds di awal training
- Log hyperparameters ke W&B
- Save checkpoints dengan naming convention yang jelas

## Commit Strategy

Prinsip: **Atomic commits, conventional format, readable git history**.

### Format

```
<type>(<scope>): <subject ≤72 char>

<body: optional, jelaskan WHY>

<footer: optional, referensi>
```

### Types

| Type | Penggunaan |
|------|-----------|
| `feat` | Fitur/kode baru |
| `fix` | Bug fix |
| `refactor` | Refactor tanpa perubahan behavior |
| `docs` | Dokumentasi |
| `test` | Test |
| `chore` | Maintenance |
| `perf` | Optimisasi performa |
| `ml` | ML-specific (training, hyperparameters, arsitektur) |
| `style` | Formatting/styling (tidak mengubah logic) |

### Scopes

`data` | `models` | `training` | `evaluation` | `pipeline` | `frontend` | `backend` | `space` | `config` | `claude` | `report` | `readme` | `deps`

### Contoh RAPI

```
feat(data): add KuzushijiDataset with stratified split
feat(models): add ResNet classifier wrapper with timm backbone
feat(frontend): add ImageDropzone component with react-dropzone
feat(backend): add FastAPI infer endpoint with rate limiting
feat(space): add Gradio app with trilingual UI
ml(resnet50): increase lr from 1e-3 to 3e-3, +0.4% balanced_acc
docs(claude): update progress with Phase 4 Week 7 completion
```

### Contoh yang TIDAK RAPI

```
update stuff
fix bugs
WIP
changes
feat: add dataset, fix typo, update readme
```

### Hal yang TIDAK Boleh di Commit

- Model weights (`*.pt`, `*.pth`, `*.onnx`)
- Dataset files (>10MB)
- Credentials (`.env`, `kaggle.json`, `*.key`)
- Node modules (`node_modules/`)
- Python virtual env (`venv/`, `.venv/`)
- Jupyter output
- Cache (`__pycache__/`, `.next/`, `.turbo/`, `wandb/`)
- OS files (`.DS_Store`, `Thumbs.db`)

## .gitignore Essentials

```
# Data
ml/data/
*.csv
*.npy
*.npz
*.h5

# Models
*.pt
*.pth
*.onnx
checkpoints/
wandb/

# Python
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/

# Node
node_modules/
.next/
.turbo/
dist/
build/

# Secrets
.env
.env.local
.env.*.local
*.key
kaggle.json

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

## Testing Strategy

- **Python Unit tests** (pytest): Data loaders, transforms, metrics, pipeline functions
- **Python Integration tests**: Full training loop on small subset
- **TypeScript Unit tests** (Vitest): Utility functions, custom hooks
- **TypeScript Component tests** (React Testing Library): Critical components
- **Coverage target**: 60% untuk critical paths
- **CI**: GitHub Actions (free tier 2000 min/month)

---

## MANUAL PROCESS GUIDE

Section ini berisi step-by-step detail untuk semua proses manual yang **tidak bisa dilakukan Claude Code**.

### 1. Setup Kaggle API

**Kapan**: Sebelum Phase 1 & 2 (download datasets)

**Langkah**:
1. Buka https://www.kaggle.com, register/login
2. Klik foto profil → "Account"
3. Scroll ke section "API" → klik "Create New API Token"
4. File `kaggle.json` ter-download
5. Pindahkan:
   - Windows: `C:\Users\{YourUsername}\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json`
6. Set permissions (macOS/Linux): `chmod 600 ~/.kaggle/kaggle.json`
7. Install: `pip install kaggle`
8. Test: `kaggle datasets list`

### 2. Setup Weights & Biases

**Kapan**: Sebelum Phase 1 (experiment tracking)

**Langkah**:
1. Buka https://wandb.ai, sign up
2. Foto profil → "User Settings" → "API Keys"
3. Copy API key
4. Install: `pip install wandb`
5. Login: `wandb login`, paste key
6. Create project "kobun-classification" di web UI
7. Ulangi untuk "kobun-detection" sebelum Phase 2

### 3. Setup HuggingFace

**Kapan**: Sebelum upload model pertama (Phase 1)

**Langkah**:
1. Buka https://huggingface.co, sign up
2. Foto profil → "Settings" → "Access Tokens"
3. Klik "New token":
   - Name: `kobun-ml`
   - Role: **Write**
4. Copy token (hanya muncul sekali)
5. Simpan di `.env`: `HF_TOKEN=hf_xxx`
6. Install: `pip install huggingface_hub`
7. Login: `huggingface-cli login`, paste token

### 4. Create HuggingFace Hub Repos

**Kapan**: Sebelum training complete Phase 1 & 2

**Langkah**: Untuk setiap model (total 4):
1. Akses https://huggingface.co/new
2. Buat repo:
   - `kobun-classifier-baseline-cnn`
   - `kobun-classifier-resnet50`
   - `kobun-classifier-vit`
   - `kobun-detector-yolov8`
3. Setting: License `cc-by-sa-4.0`, Private (ubah ke public setelah Phase 5)

### 5. Setup Gemini API

**Kapan**: Sebelum Phase 2 (translation layer)

**Langkah**:
1. Buka https://aistudio.google.com, login Google
2. Sidebar → "Get API key" → "Create API key"
3. Copy key (format `AIza...`)
4. Simpan di `.env`: `GEMINI_API_KEY=AIza...`
5. Test dengan Python call

### 6. Create GitHub Monorepo

**Kapan**: Sebelum commit pertama Phase 1

**Langkah**:
1. Akses https://github.com/new
2. Repo name: `kobun`
3. Public atau Private (bisa ubah nanti)
4. JANGAN centang README/gitignore/license (akan dibuat)
5. Clone: `git clone https://github.com/{user}/kobun.git`
6. Set identity (kalau belum): `git config user.name`, `user.email`

### 7. Setup Local Development Environment

**Kapan**: Sebelum task pertama Phase 1

**Python**:
1. Install Python 3.10+ (python.org, brew, atau apt)
2. Verify: `python --version`

**Node.js**:
1. Install Node.js 20+ (nodejs.org atau nvm)
2. Install pnpm: `npm install -g pnpm`
3. Verify: `node --version`, `pnpm --version`

**Git**:
- Windows: https://git-scm.com/download/win
- macOS: `brew install git`
- Linux: `sudo apt install git`

**VS Code**:
- Download dari https://code.visualstudio.com
- Extensions: Python, Pylance, Jupyter, TypeScript, Prettier, ESLint, Ruff, Tailwind CSS IntelliSense

**Project Setup**:
```bash
cd kobun

# Python (ML + Backend)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ml/requirements.txt

# Node (Frontend)
cd apps/web
pnpm install

# Copy env
cp .env.example .env  # edit dengan API keys
```

### 8. Setup Kaggle Notebook (Training)

**Kapan**: Sebelum training model pertama

**Langkah**:
1. Login https://www.kaggle.com, klik "Create" → "New Notebook"
2. Settings sidebar kanan:
   - Accelerator: **GPU P100**
   - Persistence: Files only
3. Upload dataset via "Add data"
4. Setup secrets: "Add-ons" → "Secrets"
   - `WANDB_API_KEY`
   - `HF_TOKEN`
   - `GEMINI_API_KEY`
5. Load secrets di notebook:
   ```python
   from kaggle_secrets import UserSecretsClient
   secrets = UserSecretsClient()
   wandb_key = secrets.get_secret("WANDB_API_KEY")
   ```
6. Background execution: "Save Version" → "Save & Run All (Commit)"

### 9. Setup Vercel Account

**Kapan**: Sebelum Phase 4 Week 7

**Langkah**:
1. Buka https://vercel.com, sign up (recommend via GitHub)
2. Grant Vercel access ke repo `kobun`
3. Klik "Add New..." → "Project"
4. Import `kobun` repo
5. **Configure**:
   - Framework Preset: Next.js
   - **Root Directory**: `apps/web` (PENTING!)
   - Build Command: `pnpm build` (default)
   - Install Command: `pnpm install` (default)
   - Output Directory: `.next` (default)
6. Environment Variables:
   - `HF_SPACE_URL` = `https://{your_hf_user}-kobun.hf.space`
   - `BACKEND_VERSION` = `2.0.0`
7. Klik "Deploy"
8. Monitor build logs
9. Setelah deploy, dapat URL `kobun-{username}.vercel.app`

### 10. Deploy to HuggingFace Space

**Kapan**: Phase 3

**Langkah**:
1. Akses https://huggingface.co/new-space
2. Settings:
   - Owner: {username}
   - Space name: `kobun`
   - License: `cc-by-sa-4.0`
   - SDK: **Gradio**
   - Hardware: **CPU basic (free)**
   - Visibility: Public
3. Klik "Create Space"
4. Clone Space repo ke folder lokal terpisah:
   ```bash
   cd /tmp
   git clone https://huggingface.co/spaces/{user}/kobun kobun-space-remote
   ```
5. Copy isi `apps/space/` dari main project:
   ```bash
   cp -r /path/to/kobun/apps/space/* kobun-space-remote/
   ```
6. Commit + push:
   ```bash
   cd kobun-space-remote
   git add .
   git commit -m "feat: initial Gradio app with trilingual UI"
   git push
   ```
7. Monitor build di HF Space page (5-10 min first time)

### 11. Setup HF Space Secrets

**Kapan**: Phase 3, setelah Space deployed

**Langkah**:
1. Akses `https://huggingface.co/spaces/{user}/kobun/settings`
2. Scroll ke "Repository secrets"
3. "New secret":
   - Name: `GEMINI_API_KEY`
   - Value: paste key
4. Space auto-rebuild dengan secret tersedia

### 12. Setup GitHub Action for Auto-sync (Optional)

**Kapan**: Setelah Phase 3 stabil, untuk automate HF Space sync

**Langkah**:
1. Di HuggingFace: Settings → Access Tokens → create new token dengan "Write access to repos"
2. Di GitHub repo: Settings → Secrets and variables → Actions → "New repository secret"
3. Add secrets:
   - `HF_TOKEN` = token HuggingFace
   - `HF_USERNAME` = username HuggingFace
4. Create `.github/workflows/space-sync.yml` (Claude Code bisa bantu buat)

### 13. Install Pandoc

**Kapan**: Phase 5 (generate PDF report)

**Langkah**:
- Windows: Download installer dari pandoc.org
- macOS: `brew install pandoc mactex`
- Linux: `sudo apt install pandoc texlive-xetex texlive-fonts-recommended`
- Test: `pandoc --version`

### 14. Setup Blog Platforms

**Kapan**: Phase 5

**Medium**:
1. Buka https://medium.com, sign up
2. Stories → "Write a story"
3. Untuk reach lebih luas, submit ke publication ML (Towards Data Science, Better Programming)

**Dev.to**:
1. https://dev.to, sign up dengan GitHub
2. "Create Post", paste markdown, add tags

### 15. Record Demo Video dengan OBS

**Kapan**: Phase 5

**Langkah**:
1. Download OBS dari https://obsproject.com
2. Setup:
   - Sources → "+", pilih "Window Capture" / "Screen Capture"
   - Pilih browser window dengan Kobun
3. Recording:
   - 0-10s: Problem statement + hero image
   - 10-40s: Live demo custom UI + HF Space demo
   - 40-60s: Architecture overview
   - 60-90s: CTA
4. Edit (optional): DaVinci Resolve / iMovie

### 16. Upload Demo Video to YouTube

**Kapan**: Setelah recording Phase 5

**Langkah**:
1. Login https://studio.youtube.com
2. "Create" → "Upload videos"
3. Settings:
   - Title: "Kobun: AI Reading Classical Japanese Documents (Kuzushiji OCR)"
   - Description: link GitHub, Vercel demo, HF Space, blog post
   - Visibility: **Unlisted**
   - Audience: "Not for kids"
4. Tags: machine learning, AI, japan, kuzushiji, computer vision, deep learning, fullstack
5. Copy shareable link, tambahkan di README dan report

---

# RINGKASAN KEPUTUSAN KUNCI

| Keputusan | Pilihan | Alasan |
|-----------|---------|--------|
| Nama project | **Kobun** (古文) | Pararel dengan Kioku, bermakna tepat |
| Arsitektur | **3-tier full-stack (Frontend + Backend + ML Server)** | Premium UI + scalable + portfolio value max |
| Repository | **Monorepo** (`apps/web/`, `apps/space/`, `ml/`) | Single source of truth, easier discovery |
| Frontend | **Next.js 15 + React 19 + shadcn/ui + Tailwind v4 + Framer Motion** | Industry standard 2026, konsisten dengan Kioku |
| Backend | **FastAPI Python di Vercel Serverless Functions** | Lightweight proxy, free tier, tight integration |
| ML Server | **HuggingFace Spaces (Gradio)** dual role (demo + API) | Public ML demo + API endpoint |
| ML Framework | **PyTorch 2.x + timm + Ultralytics** | Industry standard, ecosystem terbesar |
| Dataset | Kuzushiji-49 + Kaggle Kuzushiji Recognition | Open source, labeled, established |
| Classification | **ResNet-50 + ViT** ensemble | Solid baseline + modern architecture |
| Detection | **YOLOv8/11-medium** | Industry standard |
| Training GPU | Kaggle Notebooks (P100) | 30h/week free, background execution |
| Experiment Tracking | Weights & Biases | Free tier generous |
| Model Registry | HuggingFace Hub | Unlimited public |
| i18n Frontend | **next-intl (JP default, EN, ID)** | Modern, type-safe, RSC-compatible |
| i18n ML Server | **Python dict-based (JP default, EN, ID)** | Simple, effective untuk Gradio |
| **Storage Pattern** | **Zero-storage (stateless across all tiers)** | Privacy + $0 cost + compliance |
| **Styling Philosophy** | **Scholarly Tech Fusion** | Distinct dari Kioku, serius + modern |
| **Comment Style** | **Minimalis, WHY bukan WHAT** | Readable, less noise |
| **Emoji di Source** | **DILARANG** | Professional code |
| **Commit Strategy** | **Atomic commits, conventional format** | Git history readable |
| Total Biaya | **$0/bulan** | Semua tools pada free tier |
| Estimasi Waktu | **9-12 minggu** part-time | Realistic dengan full-stack scope |

---

> **Dokumen ini adalah fondasi project Kobun. Setiap keputusan di sini menjadi acuan saat implementation dimulai. Kobun adalah full-stack ML project standalone dalam portfolio ecosystem bersama Kioku dan future Japan projects.**

> **Versi 2.0 (Final — Full-Stack Architecture) — Ready for Implementation**
