# CLAUDE.md — Kobun (古文) Project

> Full-stack ML system untuk membaca dokumen Jepang kuno (kuzushiji).  
> Monorepo: Next.js + FastAPI (Vercel) + Gradio ML Server (HuggingFace Space).  
> PyTorch + YOLO + ONNX + Gemini (trilingual JP/EN/ID).  
> Full spec: `/designs/kobun-project-spec.md`

## Architecture

```
User browser
    ↓
Frontend (Next.js 15 + React 19 + shadcn/ui)
  Deploy: Vercel
    ↓ HTTP API
Backend (FastAPI Python Serverless)
  Deploy: Vercel Serverless Functions
  Role: Lightweight proxy + rate limiting
    ↓ HTTP API
ML Server (Gradio + ONNX models)
  Deploy: HuggingFace Spaces (free CPU)
  Role: Public ML demo + API endpoint
```

## Tech Stack

### Frontend
- **Framework**: Next.js 15 (App Router) + React 19 + TypeScript 5
- **Styling**: Tailwind CSS v4 + shadcn/ui + Framer Motion
- **State**: Zustand (client) + TanStack Query v5 (server)
- **i18n**: next-intl (JP default, EN, ID)
- **Theme**: next-themes (dark/light mode)
- **Deploy**: Vercel Hobby

### Backend
- **Framework**: FastAPI + Pydantic v2
- **HTTP Client**: httpx (async)
- **Deploy**: Vercel Serverless Functions (Python runtime)
- **Role**: Lightweight proxy (validation + rate limit + forward)

### ML Server (HuggingFace Space)
- **Framework**: Gradio 4.x (UI + auto-generated API)
- **Inference**: ONNX Runtime (INT8 quantized)
- **Translation**: Gemini 2.5 Flash-Lite
- **i18n**: Python dict-based (JP default, EN, ID)
- **Deploy**: HuggingFace Spaces (free CPU 16GB)

### ML Training
- **Framework**: PyTorch 2.x + timm + Ultralytics YOLOv8/11
- **Augmentations**: Albumentations
- **Tracking**: Weights & Biases (free tier)
- **GPU**: Kaggle Notebooks (P100, 30h/week, background execution)
- **Model Registry**: HuggingFace Hub (free, unlimited public)

### DevTools
- **Source Control**: Git + GitHub
- **Python**: Ruff + Black + pytest
- **TypeScript**: ESLint + Prettier + Vitest
- **Documentation**: Pandoc (PDF), Mermaid, matplotlib

## ATURAN PENTING

- SELALU baca `/designs/kobun-project-spec.md` untuk detail spec lengkap sebelum coding
- Ikuti Coding Rules di spec (no emoji di source code, comment minimalis, type hints/strict TS wajib)
- Ikuti Commit Strategy: atomic commits, conventional format (feat/fix/ml/docs/test/chore/perf/refactor/style)
- Trilingual UI & output: Japanese (default) / English / Indonesian
- Zero-storage architecture: foto user tidak pernah disimpan di disk/DB di tier manapun
- Monorepo structure: `apps/web/` + `apps/space/` + `ml/`
- Total biaya project: $0 (semua free tier)

## Repository Structure

```
kobun/
├── README.md                     # Umbrella README (bilingual EN+JP)
├── EVALUATION_REPORT.md          # Paper-style ML report
├── EVALUATION_REPORT.pdf         # Generated from markdown
├── CLAUDE.md                     # This file
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── Makefile                      # Workflow shortcuts
├── designs/
│   └── kobun-project-spec.md     # Full spec (copy of Instructions)
│
├── apps/
│   ├── web/                      # Next.js 15 frontend + Vercel FastAPI backend
│   │   ├── src/
│   │   │   ├── app/[locale]/     # Trilingual routing (ja/en/id)
│   │   │   ├── components/
│   │   │   │   ├── ui/           # shadcn/ui components
│   │   │   │   ├── layout/       # Navbar, Footer
│   │   │   │   ├── upload/       # ImageDropzone
│   │   │   │   ├── results/      # ResultsContainer, BoundingBoxOverlay
│   │   │   │   └── language/     # LanguageSwitcher
│   │   │   ├── lib/              # api.ts, i18n.ts, utils.ts
│   │   │   ├── stores/           # Zustand stores
│   │   │   ├── messages/         # next-intl translations (ja.json, en.json, id.json)
│   │   │   └── types/
│   │   ├── api/                  # Vercel Python Serverless Functions
│   │   │   ├── infer.py          # POST /api/infer
│   │   │   ├── health.py         # GET /api/health
│   │   │   └── requirements.txt
│   │   ├── public/
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── vercel.json
│   │
│   └── space/                    # Gradio ML Server (deploy ke HuggingFace)
│       ├── app.py
│       ├── i18n.py               # LOCALES dict
│       ├── requirements.txt
│       ├── README.md             # HF Space frontmatter (bilingual EN+JP)
│       └── examples/             # 10-15 public domain samples
│
├── ml/                           # ML training pipeline
│   ├── src/
│   │   ├── data/{datasets,transforms,yolo_converter}.py
│   │   ├── models/{classifier,detector}.py
│   │   ├── training/{trainer,callbacks,losses}.py
│   │   ├── evaluation/{metrics,error_analysis,visualization}.py
│   │   ├── pipeline/{inference,reading_order,translation}.py
│   │   ├── utils/{config,logging}.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── export.py
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
│   │   ├── download_data.py
│   │   ├── prepare_detection_data.py
│   │   ├── benchmark_inference.py
│   │   └── generate_report_pdf.py
│   ├── tests/
│   └── requirements.txt
│
└── .github/workflows/
    ├── frontend-ci.yml
    ├── ml-ci.yml
    └── space-sync.yml            # Auto-sync apps/space/ to HF Space
```

## Deployment Targets

| Component | Source | Deploy | URL |
|-----------|--------|--------|-----|
| Frontend | `apps/web/src/` | Vercel | `kobun-{user}.vercel.app` |
| Backend API | `apps/web/api/` | Vercel Serverless | `{frontend}/api/*` |
| ML Server | `apps/space/` | HuggingFace Spaces | `{user}-kobun.hf.space` |
| Models | `ml/` outputs | HuggingFace Hub | `huggingface.co/{user}/kobun-*` |

## Dataset Summary

- **Kuzushiji-49**: 270K images, 49 classes, 28x28 grayscale, imbalanced (CC BY-SA 4.0)
- **Kaggle Kuzushiji Recognition**: ~3,900 full-page scans + ~683K bounding boxes (CC BY-SA 4.0)
- **Unicode Translation CSV**: kuzushiji unicode → modern hiragana mapping
- Cite: Clanuwat et al. 2018

## Models Plan

| Model | Architecture | Target Metric | Training Time |
|-------|--------------|---------------|---------------|
| Baseline CNN | 3-conv + 2-FC (~100K params) | ~85% balanced acc | <10 min P100 |
| ResNet-50 | timm/resnet50 pretrained ImageNet | >96% balanced acc | ~1-2h P100 |
| ViT-Base | timm ViT-Base, patch 4x4 | >97% balanced acc | ~2-3h P100 |
| YOLOv8-medium | Ultralytics, COCO pretrained | mAP@0.5 >0.90 | ~4-6h P100 |

## Status Progress (Diupdate Berkala)

### [P0] Manual Setup & Foundation

- [ ] Kaggle account + API key (`kaggle.json` at `~/.kaggle/`)
- [ ] Weights & Biases account + API key + create project `kobun-classification`
- [ ] HuggingFace account + write token
- [ ] Create HF Hub repos: 4 model repos (baseline, resnet50, vit, yolov8)
- [ ] Gemini API key dari Google AI Studio
- [ ] GitHub repo `kobun` (monorepo)
- [ ] Vercel account + connect GitHub
- [ ] Local environment: Python 3.10+, Node.js 20+, pnpm, git, VS Code + extensions
- [ ] Virtual env + `.env` populated
- [ ] Initial monorepo structure (apps/, ml/, .github/)
- [ ] Initial commit: README draft + LICENSE + .gitignore + folder skeleton

### [P1] Classification (Week 1-2)

- [ ] `ml/src/data/datasets.py` — KuzushijiDataset class with augmentation
- [ ] `ml/src/data/transforms.py` — Augmentation pipeline (rotation, erasing, mixup)
- [ ] `ml/src/models/classifier.py` — ResNet/ViT wrapper via timm
- [ ] `ml/src/training/trainer.py` — Training loop with W&B integration
- [ ] `ml/src/training/losses.py` — Class-weighted, mixup, label smoothing
- [ ] `ml/src/evaluation/metrics.py` — Balanced accuracy, confusion matrix
- [ ] `ml/src/utils/config.py` — YAML config loader
- [ ] `ml/src/train.py` — CLI entry point
- [ ] `ml/scripts/download_data.py` — Kaggle API dataset downloader
- [ ] `ml/configs/baseline-cnn.yaml`, `resnet50-baseline.yaml`, `vit-base.yaml`
- [ ] `ml/notebooks/01_eda.ipynb` — Class distribution, sample viz, imbalance
- [ ] Train baseline CNN (sanity check)
- [ ] Train ResNet-50 with full augmentations (target >96%)
- [ ] Train ViT-Base (target >97%)
- [ ] W&B hyperparameter sweep (20-30 runs)
- [ ] Upload best models ke HF Hub dengan model cards (draft)
- [ ] `EVALUATION_REPORT.md` partial — Phase 1 section

### [P2] Detection + ML Pipeline (Week 3-5)

- [ ] Download Kaggle Kuzushiji Recognition dataset
- [ ] `ml/src/data/yolo_converter.py` — Bounding box CSV → YOLO format
- [ ] `ml/configs/yolov8-detection.yaml` — YOLO training config
- [ ] Custom anchor boxes via k-means
- [ ] Fine-tune YOLOv8/YOLO11-medium
- [ ] Evaluate detection (mAP@0.5 target >0.90)
- [ ] `ml/src/pipeline/inference.py` — Two-stage pipeline (detect → crop → classify)
- [ ] `ml/src/pipeline/reading_order.py` — Top-to-bottom, right-to-left sort
- [ ] `ml/src/pipeline/translation.py` — Gemini API wrapper trilingual (JP+EN+ID single call)
- [ ] Test pipeline on held-out pages (target character accuracy >85%)
- [ ] Upload YOLO model ke HF Hub dengan model card
- [ ] `ml/notebooks/02_detection_training.ipynb`
- [ ] `ml/notebooks/03_pipeline_demo.ipynb`
- [ ] `EVALUATION_REPORT.md` updated — Phase 2 section

### [P3] ML Server / HuggingFace Space (Week 6)

- [ ] `ml/src/export.py` — ONNX export + INT8 quantization
- [ ] Benchmark inference speed on CPU (target <3s per image)
- [ ] `apps/space/i18n.py` — Trilingual LOCALES dict (JP/EN/ID)
- [ ] `apps/space/app.py` — Gradio app with zero-storage + language switcher
- [ ] `apps/space/examples/` — 10-15 public domain sample images
- [ ] `apps/space/requirements.txt`
- [ ] `apps/space/README.md` — Bilingual EN+JP with frontmatter
- [ ] Create HuggingFace Space "kobun" (Gradio SDK, free CPU)
- [ ] Add GEMINI_API_KEY secret to HF Space
- [ ] Initial git push to Space repository
- [ ] Test inference latency on deployed Space
- [ ] Test trilingual UI transitions
- [ ] Test auto-generated API endpoint (`POST /run/predict`)
- [ ] Verify privacy notice visible dalam 3 bahasa
- [ ] Public URL: `https://{username}-kobun.hf.space`

### [P4] Custom Frontend + Backend API (Week 7-10)

#### Week 7 — Setup + Core
- [ ] Setup `apps/web/` Next.js 15 + TypeScript via `pnpm create next-app`
- [ ] Configure Tailwind CSS v4 + design tokens (Indigo/Vermillion/Beige)
- [ ] Install dan setup shadcn/ui (CLI init)
- [ ] Install Framer Motion 12, Zustand 5, TanStack Query 5
- [ ] Setup next-intl middleware untuk trilingual routing (`/ja`, `/en`, `/id`)
- [ ] Setup next-themes untuk dark/light mode
- [ ] Create `apps/web/src/messages/{ja,en,id}.json` dengan key structure
- [ ] Layout components: `Navbar`, `Footer`, `Container`, `PageWrapper`
- [ ] `LanguageSwitcher` component (dropdown JP/EN/ID)
- [ ] `ThemeToggle` component
- [ ] Setup font loading: Fraunces (display), Inter (body), Noto Serif JP, Noto Sans JP

#### Week 8 — Landing Page
- [ ] `HeroSection` component dengan animated overlay
- [ ] `SampleGallery` component (grid 10-15 thumbnails)
- [ ] `HowItWorksSection` component (3-step visualization)
- [ ] `TechnicalDetailsSection` component (model metrics showcase)
- [ ] `CTASection` component
- [ ] Footer content (links: GitHub, HF Space, paper, citation, license)
- [ ] Responsive design verification (mobile/tablet/desktop)
- [ ] Framer Motion page transitions
- [ ] SEO meta tags + Open Graph

#### Week 9 — Upload, Results, Backend
- [ ] `apps/web/api/infer.py` — FastAPI endpoint dengan Pydantic validation
- [ ] `apps/web/api/health.py` — Health check endpoint
- [ ] `apps/web/api/requirements.txt` — Backend deps (fastapi, pydantic, httpx)
- [ ] Rate limiting (in-memory sliding window, 10 req/min per IP)
- [ ] `ImageDropzone` component (react-dropzone) dengan validation
- [ ] `ImagePreview` component
- [ ] `UploadProgress` component dengan stage indicator
- [ ] `ResultsContainer` component
- [ ] `AnnotatedImage` + `BoundingBoxOverlay` (SVG, color by confidence)
- [ ] `TranscriptionCard` (modern Japanese)
- [ ] `TranslationCard` (selected language)
- [ ] `MetadataBar` + `ActionButtons` (copy/share/try again)
- [ ] TanStack Query setup untuk API calls (`useInferMutation`)
- [ ] Error handling + toast notifications

#### Week 10 — Polish + Deploy
- [ ] Test full flow end-to-end (local: frontend → backend → HF Space)
- [ ] Setup Vercel project linked to monorepo (root: `apps/web/`)
- [ ] Set Vercel env vars (`HF_SPACE_URL`, `BACKEND_VERSION`)
- [ ] First deploy ke Vercel
- [ ] Test production environment
- [ ] Verify CORS dan security headers
- [ ] Lighthouse optimization (Performance >85, A11y >95, SEO >95)
- [ ] Open Graph image, favicon, manifest.json
- [ ] Setup custom domain (optional)
- [ ] `EVALUATION_REPORT.md` Phase 3+4 section (System Architecture)

### [P5] Technical Report + Reproducibility (Week 11-12)

- [ ] `EVALUATION_REPORT.md` — All sections complete (Abstract → Acknowledgments)
- [ ] Error analysis (worst classes, confusion pairs, failure gallery)
- [ ] Generate PDF via pandoc (`make report`)
- [ ] Finalize 4 model cards on HF Hub
- [ ] `Makefile` lengkap (setup, data, train, evaluate, export, dev-frontend, dev-backend, dev-space, deploy-space, report)
- [ ] Test reproduce-from-scratch on clean environment (target: <2 jam setup to working demo)
- [ ] `ml/notebooks/04_pipeline_demo.ipynb` — End-to-end demo
- [ ] `ml/notebooks/05_error_analysis.ipynb` — Detailed error analysis
- [ ] Record demo video dengan OBS (60-90s)
- [ ] Upload video ke YouTube (Unlisted)
- [ ] Write blog post English version (Medium + Dev.to)
- [ ] Write blog post Indonesian version (optional)
- [ ] Finalize umbrella `README.md` (bilingual EN + JP, hero image, badges, all links)
- [ ] `CITATION.cff` — Citation metadata
- [ ] `CHANGELOG.md` — Version history
- [ ] Publish HF Space as public
- [ ] Flip 4 HF Hub models to public
- [ ] Setup GitHub Action `space-sync.yml` (optional, automate HF Space sync)

---

## Catatan Teknis (Diupdate Seiring Development)

_Belum ada catatan. Akan diisi saat implementasi berjalan dan ada keputusan teknis yang perlu didokumentasikan._

---

## Priority Phases

- **[P0] Manual Setup**: Akun + local env + monorepo structure (lihat Manual Process Guide di spec)
- **[P1] Classification**: Baseline CNN + ResNet-50 + ViT + model cards + Phase 1 report
- **[P2] Detection + Pipeline**: YOLO + two-stage pipeline + trilingual translation + Phase 2 report
- **[P3] ML Server**: ONNX export + Gradio app + HF Space deployment + auto API endpoint
- **[P4] Custom Frontend + Backend**: Next.js 15 + shadcn/ui + Framer Motion + FastAPI proxy + Vercel deploy
- **[P5] Report + Reproducibility**: Complete report + blog post + demo video + reproduce-from-scratch

## Evaluation Targets

| Phase | Metric | Target |
|-------|--------|--------|
| P1 Classification | Balanced accuracy ResNet-50 | >96% |
| P1 Classification | Balanced accuracy ViT | >97% |
| P2 Detection | mAP@0.5 | >0.90 |
| P2 Detection | Precision / Recall | >0.92 / >0.88 |
| P2 End-to-End | Character accuracy | >85% |
| P2 End-to-End | Sequence accuracy | >70% |
| P2 End-to-End | Reading order accuracy | >95% |
| P3 ML Server | Inference time (CPU) | <3 seconds |
| P4 Frontend | Lighthouse Performance | >85 |
| P4 Frontend | Lighthouse Accessibility | >95 |
| P4 Backend | Cold start | <5s |
| P4 Backend | Warm response (excl. inference) | <200ms |
| P4 End-to-end | Total latency | <5s per image |

## Commit Strategy Cheat Sheet

**Types**: `feat` | `fix` | `refactor` | `docs` | `test` | `chore` | `perf` | `ml` | `style`

**Format**: `<type>(<scope>): <subject ≤72 char>`

**Scopes**: `data` | `models` | `training` | `evaluation` | `pipeline` | `frontend` | `backend` | `space` | `config` | `claude` | `report` | `readme` | `deps`

**Principle**: 1 commit = 1 logical change (bukan 1 task = 1 commit). Atomic, informative, readable git history.

**Examples**:
```
feat(data): add KuzushijiDataset with stratified split
feat(models): add ResNet classifier wrapper
feat(frontend): add ImageDropzone component with react-dropzone
feat(backend): add FastAPI infer endpoint with rate limiting
feat(space): add Gradio app with trilingual UI
ml(resnet50): increase lr 1e-3→3e-3, +0.4% balanced_acc
docs(claude): update P4 week 8 progress
```

## Coding Rules

### Universal
- **Variables/functions/comments/commits**: English. **User-facing**: Trilingual (JP/EN/ID)
- **NO emoji/emoticon di source code**. Emoji hanya boleh di public README untuk visual appeal
- **Comment minimalis**: hanya saat perlu (algoritma kompleks, WHY non-obvious). Jangan comment yang redundant
- **No hardcoded secrets**: semua via env vars atau platform secrets

### Python
- **Type hints**: Wajib untuk production code (`ml/src/`, `apps/web/api/`, `apps/space/`)
- **Docstrings**: Google style, wajib untuk non-trivial functions
- **Naming**: files=snake_case | classes=PascalCase | functions=snake_case | constants=UPPER_SNAKE
- **Linting**: Ruff + Black

### TypeScript
- **Strict mode**: `strict: true` di tsconfig.json. No `any` tanpa justifikasi
- **Component pattern**: Function declaration untuk exports, props interface above
- **Naming**: files=kebab-case | components=PascalCase | functions/vars=camelCase
- **Server vs Client**: Default Server Components, "use client" only saat butuh interactivity
- **Linting**: ESLint + Prettier

### Imports Order
- Python: stdlib → third-party → local (blank line antara group)
- TypeScript: external → internal libs → components → types → styles

### Error Handling
- API responses: `{ success: boolean, data?: T, error?: { code, message } }`
- Try-catch critical paths, log with context
- Graceful fallbacks untuk user

## Reproducibility Standard

Stranger dengan:
- Python 3.10+ + Node.js 20+ + pnpm
- `git clone` repo
- Kaggle API key di `~/.kaggle/`
- W&B API key
- HF token
- Gemini API key
- Vercel account (untuk deploy frontend)

Harus bisa run:
```bash
make setup           # Install all deps (Python + Node)
make data            # Download datasets
make train-all       # Train all models (dengan GPU)
make evaluate        # Run evaluations
make export          # ONNX + quantize
make dev-frontend    # Local Next.js dev server
make dev-backend     # Local FastAPI server
make dev-space       # Local Gradio server
make deploy-space    # Sync apps/space/ to HF Space
make report          # Generate EVALUATION_REPORT.pdf
```

Reproduce semua hasil dalam <2 jam (excluding GPU training yang butuh 6-8 jam total). Training reproducibility via config YAML.

## URL Reference (Setelah Phase 5 Complete)

- **Custom Web App**: `https://kobun-{username}.vercel.app`
- **Public ML Demo**: `https://huggingface.co/spaces/{username}/kobun`
- **GitHub**: `https://github.com/{username}/kobun`
- **Models**: `https://huggingface.co/{username}/kobun-*`
- **Demo Video**: `https://youtube.com/watch?v={video_id}` (Unlisted)
- **Blog Post**: Medium + Dev.to (English + Indonesian)
- **Paper**: PDF di GitHub repo (`EVALUATION_REPORT.pdf`)
